from typing import *
import os
import json
from multiprocessing import Pool
from ._extractors import (
    extract_basic_info,
    extract_employee_info,
    extract_cbs_info,
    extract_cscf_info,
    extract_cis_info,
    extract_dev_info,
    extract_metadata,
    extract_pure_content,
    merge_all_table_infos,
)
from ._table_builder import PreprocessorTableBuilder
from tqdm import tqdm
from ._checker import PreprocessorChecker
from ._pdf2txt import process_all_pdfs_in_folder

module_path = os.path.dirname(__file__)


class Preprocessor:
    def __init__(
        self,
        ctx_dir: str = os.path.join("resources", "processed_data"),
        doc_dir_name="docs",
    ) -> None:
        self.ctx_dir = ctx_dir
        self.doc_dir = os.path.join(ctx_dir, doc_dir_name)
        self.pdf_metadata = {}
        self.table_builder = PreprocessorTableBuilder(
            ctx_dir=ctx_dir, doc_dir_name=doc_dir_name
        )
        self.checker = PreprocessorChecker(
            ctx_dir=self.ctx_dir, doc_dir_name=doc_dir_name
        )
        if not os.path.exists(ctx_dir):
            os.makedirs(ctx_dir)
        if not os.path.exists(self.doc_dir):
            os.makedirs(self.doc_dir)

    def gen_pdf_metadata(
        self, pdf_dir: str = "", target_pdf_paths=[], persist=False
    ) -> Dict:
        """
        分析pdf文件名, 并生成元数据.
        key是文件名，value是元数据
        示例:
        {
            "2020-01-21__江苏安靠智能输电工程科技股份有限公司__300617__安靠智电__2019年__年度报告.pdf": {
                "key": "2020-01-21__江苏安靠智能输电工程科技股份有限公司__300617__安靠智电__2019年__年度报告.pdf",
                "pdf_path": "/home/xxx/Desktop/advanced_qa/preprocess/./example/2020-01-21__江苏安靠智能输电工程科技股份有限公司__300617__安靠智电__2019年__年度报告.pdf",
                "company": "江苏安靠智能输电工程科技股份有限公司",
                "code": "300617",
                "abbr": "安靠智电",
                "year": "2019年"
            }
        }
        """
        ds = {}
        if pdf_dir:
            target_pdf_paths = get_all_files_paths_in_dir(pdf_dir)
        assert (
            len(target_pdf_paths) > 0
        ), "no pdf files found, please set the 'pdf_dir' or 'target_pdf_paths' parameter correctly"

        for pdf_path in target_pdf_paths:
            # get the last element of the path
            name = os.path.basename(pdf_path)
            split = name.split("__")
            ds[name] = {
                "key": name,
                "pdf_path": pdf_path,
                "company": split[1],
                "code": split[2],
                "abbr": split[3],
                "year": split[4],
            }
        if persist:
            with open(
                os.path.join(self.ctx_dir, "pdf_metadata.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(ds, f, ensure_ascii=False, indent=4)
        self.pdf_metadata = ds
        return ds

    def extract_pdf_text(
        self,
        xpdf_path: str,
        num_processors=4,
    ):
        """
        将pdf解析为纯文本jsonl并保存
        示例格式:
            {"page": 1, "text": "blablabla"}
            {"page": 2, "text": "blablabla"}
            {"page": 3, "text": "blablabla"}
            ...
        """

        with Pool(processes=num_processors) as pool:
            params = [
                (k, v["pdf_path"], self.doc_dir, xpdf_path)
                for i, (k, v) in enumerate(self.pdf_metadata.items())
            ]

            results = [
                pool.starmap_async(
                    extract_pure_content,
                    [param],
                )
                for param in params
            ]

            for r in tqdm(results, total=len(params), desc="extracting pure content"):
                r.wait()

    def extract_pdf_tables(self, num_processors=4):
        """
        从pdf文件中抽取各类表信息
        保存所有表单独的中间结果:
            basic_info.json
            employee_info.json
            cbs_info.json
            cscf_info.json
            cis_info.json
            dev_info.json
            metadata.json
        保存所有表合并后的结果: merged.json
        """
        args = [(v, self.doc_dir) for i, (k, v) in enumerate(self.pdf_metadata.items())]
        # metadata
        with Pool(processes=num_processors) as pool:
            results = [pool.starmap_async(extract_metadata, [arg]) for arg in args]
            for r in tqdm(results, total=len(args), desc="extracting metadata"):
                r.wait()
        # basic_info
        with Pool(processes=num_processors) as pool:
            results = [pool.starmap_async(extract_basic_info, [arg]) for arg in args]
            for r in tqdm(results, total=len(args), desc="extracting basic_info"):
                r.wait()
        # employee_info
        with Pool(processes=num_processors) as pool:
            results = [pool.starmap_async(extract_employee_info, [arg]) for arg in args]
            for r in tqdm(results, total=len(args), desc="extracting employee_info"):
                r.wait()
        # cbs_info
        with Pool(processes=num_processors) as pool:
            results = [pool.starmap_async(extract_cbs_info, [arg]) for arg in args]
            for r in tqdm(results, total=len(args), desc="extracting cbs_info"):
                r.wait()
        # cscf_info
        with Pool(processes=num_processors) as pool:
            results = [pool.starmap_async(extract_cscf_info, [arg]) for arg in args]
            for r in tqdm(results, total=len(args), desc="extracting cscf_info"):
                r.wait()
        # cis_info
        with Pool(processes=num_processors) as pool:
            results = [pool.starmap_async(extract_cis_info, [arg]) for arg in args]
            for r in tqdm(results, total=len(args), desc="extracting cis_info"):
                r.wait()
        # dev_info
        with Pool(processes=num_processors) as pool:
            results = [pool.starmap_async(extract_dev_info, [arg]) for arg in args]
            for r in tqdm(results, total=len(args), desc="extracting dev_info"):
                r.wait()

        merge_all_table_infos(self.doc_dir)

    def check_extractions(
        self, persist_err_report=False, delete_malformed_pure_content=False
    ) -> int:
        """
        检查抽取结果
        return: 抽取错误的pdf数量
        """

        self.checker.check_pure_content(
            persist_err_report=persist_err_report,
            delete_malformed_pure_content=delete_malformed_pure_content,
        )
        self.checker.check_tables(persist_err_report=persist_err_report)
        return self.checker.err_cnt

    def build_table(self, persist=False):
        """
        基于所有抽取结果，建立大宽表
        """
        self.table_builder.build_table(persist=persist)

    def parse_all_pdf_2_txt(
        self, target_pdf_paths: List[str], num_processors=4, skip_existing=True
    ):
        """
        解析所有pdf文件, 生成txt文件
        """
        args = [([a_path], self.ctx_dir, skip_existing) for a_path in target_pdf_paths]
        with Pool(processes=num_processors) as pool:
            results = [
                pool.starmap_async(process_all_pdfs_in_folder, [arg]) for arg in args
            ]
            for r in tqdm(results, total=len(args), desc="converting pdfs to txts"):
                r.wait()


"""
Tests
"""


def get_all_files_paths_in_dir(dir_path: str):
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path)]


test_ctx_dir = os.path.join(module_path, "test_temp")
test_example_dir = os.path.join(module_path, "example")


def test_gen_pdf_metadata():
    p = Preprocessor(ctx_dir=test_ctx_dir)
    p.gen_pdf_metadata(
        pdf_dir=test_example_dir,
        persist=True,
    )


def test_extract_pdf_text():
    p = Preprocessor(ctx_dir=test_ctx_dir)
    p.gen_pdf_metadata(
        pdf_dir=test_example_dir,
        persist=False,
    )
    p.extract_pdf_text(xpdf_path=os.path.join(module_path, "./xpdf/bin64"))


def test_extract_pdf_tables():
    p = Preprocessor(ctx_dir=test_ctx_dir)
    p.gen_pdf_metadata(
        pdf_dir=test_example_dir,
        persist=False,
    )
    p.extract_pdf_tables()


def test_build_table():
    p = Preprocessor(ctx_dir=test_ctx_dir)
    err_cnt = p.check_extractions(persist_err_report=True)
    print(f"{err_cnt} pdfs have extraction errors")
    p.build_table(persist=True)

import os
import shutil
from ._browser import PreprocessorBrowser
from typing import *
from ._browser import PdfMetadata
from tqdm import tqdm
from loguru import logger

module_path = os.path.dirname(__file__)


class PreprocessorChecker(object):
    def __init__(
        self,
        ctx_dir: str,
        doc_dir_name: str = "docs",
        copy_error_pdf=True,
        err_dir="",
    ):
        self.copy_error_pdf = copy_error_pdf
        self.ctx_dir = ctx_dir
        self.doc_dir = os.path.join(ctx_dir, doc_dir_name)
        self.err_dir = err_dir
        self.err_cnt = 0
        self.browser = PreprocessorBrowser(doc_dir=self.doc_dir)
        self.err_report: List[str] = []
        if self.err_dir == "":
            self.err_dir = os.path.join(self.ctx_dir, "err_pdf")
        if copy_error_pdf and os.path.exists(self.err_dir):
            shutil.rmtree(self.err_dir)

    def check_pure_content(
        self,
        target_file_name="pure_content.txt",
        persist_err_report=False,
        delete_malformed_pure_content=False,
    ):
        items = self.browser.get_pure_contents(target_file_name=target_file_name)
        for item in tqdm(items, total=len(items), desc="checking pure content"):
            if len(item.text_lines) == 0:
                self.export_err_file(item.metadata, "pure content malformed")
                if delete_malformed_pure_content:
                    logger.info(
                        f"deleting malformed pure content file: {item.metadata.pdf_path}"
                    )
                    os.remove(item.path)
        if persist_err_report:
            with open(
                os.path.join(self.ctx_dir, "pure_content_err_report.txt"), "w"
            ) as f:
                for line in self.err_report:
                    f.write(line + "\n")

    def check_tables(
        self, target_file_name="merged.json", persist_err_report=False
    ) -> List[str]:
        items = self.browser.get_tables(target_file_name=target_file_name)
        for item in tqdm(items, total=len(items), desc="checking tables"):
            reasons = []
            if len(item.basic_info) == 0:
                reasons.append("basic info missing")
            if len(item.employee_info) == 0:
                reasons.append("employee info missing")
            if len(item.cbs_info) == 0:
                reasons.append("cbs info missing")
            if len(item.cscf_info) == 0:
                reasons.append("cscf info missing")
            if len(item.cis_info) == 0:
                reasons.append("cis info missing")
            if len(item.dev_info) == 0:
                reasons.append("dev info missing")
            if len(reasons) > 0:
                self.export_err_file(item.metadata, ",".join(reasons))
        if persist_err_report:
            with open(os.path.join(self.ctx_dir, "table_err_report.txt"), "w") as f:
                for line in self.err_report:
                    f.write(line + "\n")

        return self.err_report

    def export_err_file(self, metadata: PdfMetadata, reason: str):
        self.err_report.append(f"reason: {reason}, path: {metadata.pdf_path}")
        if not self.copy_error_pdf:
            return
        if not os.path.exists(self.err_dir):
            os.mkdir(self.err_dir)
        shutil.copy(metadata.pdf_path, self.err_dir)
        self.err_cnt += 1


def test_checker():
    checker = PreprocessorChecker(ctx_dir=os.path.join(module_path, "test_temp"))
    checker.check_tables(target_file_name="merged.json", persist_err_report=True)
    checker.check_pure_content(
        target_file_name="pure_content.txt", persist_err_report=True
    )
    print(f"{checker.err_cnt} errors found")

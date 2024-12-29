import os
import json
from dataclasses import dataclass, asdict
from typing import *
import pandas as pd
from ._table_builder_util import *


module_path = os.path.dirname(__file__)


@dataclass
class PdfMetadata:
    key: str
    pdf_path: str
    company: str
    code: str
    abbr: str
    year: str


@dataclass
class PdfPureContent:
    metadata: PdfMetadata
    text_lines: List[str]
    path: str


@dataclass
class PdfTable:
    metadata: PdfMetadata
    basic_info: List[str]
    employee_info: List[str]
    cbs_info: List[str]
    cscf_info: List[str]
    cis_info: List[str]
    dev_info: List[str]


dir_pattern = r"^(\d{4}-\d{2}-\d{2})__([^_]+)__\d+__([^_]+)__\d{4}年__年度报告\.pdf$"


class PreprocessorBrowser(object):
    def __init__(
        self,
        doc_dir: str,
    ):
        self.doc_dir = doc_dir

    def get_pure_contents(
        self, target_file_name="pure_content.txt", pdf_name=""
    ) -> List[PdfPureContent]:
        results = []
        for root, dirs, files in os.walk(self.doc_dir):
            for dir in dirs:
                if pdf_name and dir != pdf_name:
                    continue
                match = re.match(dir_pattern, dir)
                if not match:
                    continue
                meta_data_path = os.path.join(root, dir, "metadata.json")
                if not os.path.exists(meta_data_path):
                    logger.warning(f"{meta_data_path} does not exist")
                    continue
                with open(meta_data_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                pure_content_path = os.path.join(root, dir, target_file_name)
                if not os.path.exists(pure_content_path):
                    logger.warning(f"{pure_content_path} does not exist")
                    continue
                with open(pure_content_path, "r", encoding="utf-8") as f:
                    text_lines = []
                    try:
                        lines = f.readlines()
                        text_lines = [json.loads(line) for line in lines]
                        text_lines = sorted(text_lines, key=lambda x: x["page"])
                    except Exception as e:
                        logger.error(f"error getting pure content: {e}")
                        results.append(
                            PdfPureContent(
                                PdfMetadata(**metadata), [], pure_content_path
                            )
                        )
                        continue
                    results.append(
                        PdfPureContent(
                            PdfMetadata(**metadata), text_lines, pure_content_path
                        )
                    )
        return results

    def get_tables(self, target_file_name="merged.json") -> List[PdfTable]:
        results = []
        for root, dirs, files in os.walk(self.doc_dir):
            for dir in dirs:
                match = re.match(dir_pattern, dir)
                if not match:
                    continue
                path = os.path.join(root, dir, target_file_name)
                if not os.path.exists(path):
                    logger.warning(f"{path} does not exist")
                    with open(
                        os.path.join(root, dir, "metadata.json"), "r", encoding="utf-8"
                    ) as f:
                        md = json.load(f)
                        results.append(
                            PdfTable(
                                PdfMetadata(**md),
                                {},
                                {},
                                {},
                                {},
                                {},
                                {},
                            )
                        )
                    continue
                with open(
                    path,
                    "r",
                    encoding="utf-8",
                ) as f:
                    target = json.load(f)

                    bi = target["basic_info"]
                    ei = target["employee_info"]
                    cbs = target["cbs_info"]
                    cscf = target["cscf_info"]
                    cis = target["cis_info"]
                    dev = target["dev_info"]
                    results.append(
                        PdfTable(
                            PdfMetadata(**target["metadata"]),
                            bi,
                            ei,
                            cbs,
                            cscf,
                            cis,
                            dev,
                        )
                    )
        return results

    def get_tables_as_df(self, target_file_name="merged.json") -> pd.DataFrame:
        tables = self.get_tables(target_file_name=target_file_name)
        pdf_tables_dicts = [asdict(item) for item in tables]
        df = pd.json_normalize(pdf_tables_dicts)
        return df

    def get_pure_content_as_df(
        self, target_file_name="pure_content.txt"
    ) -> pd.DataFrame:
        contents = self.get_pure_contents(target_file_name=target_file_name)
        pdf_contents_dicts = [asdict(item) for item in contents]
        df = pd.json_normalize(pdf_contents_dicts)
        return df

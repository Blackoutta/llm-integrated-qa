import pandas as pd
import os
import json
from pydantic import BaseModel, ConfigDict
from typing import *
from loguru import logger
import sqlite3
import numpy as np
from pathlib import Path

file_dir = os.path.dirname(__file__)


class DataLoader(BaseModel):
    ctx_dir: Path
    inference_dir: Path
    pdf_metadata_map: dict = None
    company_table: pd.DataFrame = None
    classfication_map: Dict[int, str] = None
    keywords_map: Dict[int, List[str]] = None
    nl2sql_map: Dict[int, str] = None
    sql_cursor: sqlite3.Cursor = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def load_pdf_metadata_map(self, data: dict = None) -> Dict:
        if self.pdf_metadata_map is not None:
            return self.pdf_metadata_map
        if data is not None:
            self.pdf_metadata_map = data
            return self.pdf_metadata_map
        with open(os.path.join(self.ctx_dir, "pdf_metadata.json"), "r") as f:
            self.pdf_metadata_map = json.load(f)
            return self.pdf_metadata_map

    def load_company_table(
        self,
        data: pd.DataFrame = None,
        remove_column_prefix=False,
        reset=True,
    ) -> pd.DataFrame:
        if reset:
            self.company_table = None

        if self.company_table is not None:
            return self.company_table
        if data is not None:
            self.company_table = data
            return self.company_table
        df = pd.read_csv(
            os.path.join(self.ctx_dir, "CompanyTable.csv"), sep="\t", encoding="utf-8"
        )
        if remove_column_prefix:
            df.columns = [
                col.replace(col, col.split(".")[1]) if len(col.split(".")) > 1 else col
                for col in df.columns
            ]
        col_names = [col for col in df.columns]
        col_names = ["- " + cn for cn in col_names]
        col_names = "\n".join(col_names)

        self.company_table = df
        return self.company_table

    def load_classification_map(self, data: dict = None) -> Dict:
        if self.classfication_map is not None:
            return self.classfication_map
        if data is not None:
            self.classfication_map = data
            return self.classfication_map
        m = {}
        with open(os.path.join(self.inference_dir, "classification.jsonl"), "r") as f:
            for line in f.readlines():
                entry = json.loads(line)
                m[entry["id"]] = entry["class"]
        self.classfication_map = m
        logger.debug(f"loaded {len(m)} classification")
        return self.classfication_map

    def load_keywords_map(self, data: dict = None) -> Dict:
        if self.keywords_map is not None:
            return self.keywords_map
        if data is not None:
            self.keywords_map = data
            return self.keywords_map
        m = {}
        with open(os.path.join(self.inference_dir, "keywords.jsonl"), "r") as f:
            for line in f.readlines():
                entry = json.loads(line)
                m[entry["id"]] = entry["keywords"]
        self.keywords_map = m
        return self.keywords_map

    def load_nl2sql_map(self, data: dict = None) -> Dict[int, str]:
        if self.nl2sql_map is not None:
            return self.nl2sql_map
        if data is not None:
            self.nl2sql_map = data
            return self.nl2sql_map
        m = {}
        with open(os.path.join(self.inference_dir, "nl2sql.jsonl"), "r") as f:
            for line in f.readlines():
                entry = json.loads(line)
                m[entry["id"]] = entry["sql"]
        self.nl2sql_map = m
        return self.nl2sql_map

    def find_company_table_data(
        self, company, years: List[Union[int, str]]
    ) -> pd.DataFrame:
        """
        return example:
        # [(table name, row_year, column name, row_value)]
        """
        # convert years to list of ints
        years = [int(year) for year in years]

        df = self.load_company_table()
        rows = df.query(f"`公司全称` == '{company}' and `年份`.isin({years})")
        tuples = []
        for index, row in rows.iterrows():
            for col_name in rows.columns:
                split = col_name.split(".")
                if len(split) == 1:
                    table_name = "no_table"
                    key = split[0]
                else:
                    table_name = split[0]
                    key = split[1]
                value = row[col_name]
                tuples.append((table_name, str(row["年份"]), key, value))
        return tuples

    def load_pdf_pure_text_alltxt(self, key: str):
        key = key.replace(".pdf", "")
        text_lines = []
        text_path = os.path.join(
            self.ctx_dir, "alltxts", "{}.txt".format(os.path.splitext(key)[0])
        )
        if not os.path.exists(text_path):
            logger.warning("{} not exists".format(text_path))
            return text_lines
        with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            raw_lines = [json.loads(line) for line in lines]
            for line in raw_lines:
                if "type" not in line or "inside" not in line:
                    continue
                if len(line["inside"].replace(" ", "")) == 0:
                    continue
                if line["type"] in ["页脚", "页眉"]:
                    continue
                if line["type"] == "text":
                    text_lines.append(line)
                elif line["type"] == "excel":
                    try:
                        row = eval(line["inside"])
                        line["inside"] = "\t".join(row)
                        text_lines.append(line)
                    except:
                        logger.warning("Invalid line {}".format(line))
                else:
                    logger.warning("Invalid line {}".format(line))

            text_lines = sorted(text_lines, key=lambda x: x["allrow"])

            if len(text_lines) == 0:
                logger.warning("{} is empty".format(text_path))

        return text_lines

    def load_pdf_pages(self, key: str):
        all_lines = self.load_pdf_pure_text_alltxt(key)
        pages = []
        if len(all_lines) == 0:
            return pages
        current_page_id = all_lines[0]["page"]
        current_page = []
        for line in all_lines:
            if line["page"] == current_page_id:
                current_page.append(line)
            else:
                pages.append("\n".join([t["inside"] for t in current_page]))
                current_page_id = line["page"]
                current_page = [line]
        pages.append("\n".join([t["inside"] for t in current_page]))
        return pages

    def load_sql_search_cursor(self) -> sqlite3.Cursor:
        """
        将dataframe转换为关系型数据库，方便查询
        """
        if self.sql_cursor is not None:
            return self.sql_cursor

        conn = sqlite3.connect(":memory:")

        df = self.load_company_table(remove_column_prefix=True)

        dtypes = {}
        for col in df.columns:
            num_count = 0
            tot_count = 0
            for v in df[col]:
                if v == "NULLVALUE":
                    continue
                tot_count += 1
                try:
                    number = float(v)
                except ValueError:
                    continue
                num_count += 1
            if tot_count > 0 and num_count / tot_count > 0.5:
                df[col] = (
                    df[col]
                    .apply(lambda t: DataLoader.col_to_numeric(t))
                    .replace([np.inf, -np.inf], np.nan)
                )
                dtypes[col] = "REAL"
            else:
                dtypes[col] = "TEXT"

        dtypes["年份"] = "TEXT"
        df.to_sql(name="company_table", con=conn, if_exists="replace", dtype=dtypes)

        cursor = conn.cursor()

        self.sql_cursor = cursor

        return cursor

    @staticmethod
    def col_to_numeric(t):
        try:
            value = float(t)
            if value > 2**63 - 1:
                return np.nan
            elif int(value) == value:
                return int(value)
            else:
                return float(t)
        except:
            return np.nan

    # deprecated
    def exec_sql_v1(self, sql) -> Tuple[str, str]:
        sql_cursor = self.load_sql_search_cursor()
        answer = ""
        try:
            result = sql_cursor.execute(sql).fetchall()
            rows = []
            for row in result[:50]:
                vals = []
                for val in row:
                    try:
                        num = float(val)
                        vals.append("{:.2f}元{:.0f}个{:.0f}家".format(num, num, num))
                    except:
                        vals.append(val)
                rows.append(",".join(map(str, vals)))
            answer += ";".join(rows)
        except Exception as e:
            return "", str(e)

        return answer, None

    def exec_sql_v2(self, sql) -> Tuple[dict, str]:
        sql_cursor = self.load_sql_search_cursor()
        try:
            result = sql_cursor.execute(sql).fetchall()
            rows = []
            for row in result:
                if len(row) == 1:
                    rows.append(f"{row[0]}")
                elif len(row) == 2:
                    rows.append(":".join([f"{val}" for val in row]))
                else:
                    rows.append(",".join([f"{val}" for val in row]))
            result = "\n".join(rows)
            return {"result": result, "executed_sql": sql}, None
        except Exception as e:
            return {}, str(e)

    def exec_sql(self, sql) -> Tuple[dict, str]:
        return self.exec_sql_v2(sql)


def test_exec_sql():
    loader = DataLoader(
        ctx_dir=Path(Path(file_dir).parent, "resources/processed_data").as_posix(),
        inference_dir=Path(Path(file_dir).parent, "resources/inferenced").as_posix(),
    )
    result, err = loader.exec_sql(
        "select 公司全称, 营业收入 from company_table where 注册地址 like '%宁波%' and 年份 = '2020' ORDER BY 营业收入 DESC LIMIT 3;"
    )
    print(f"result: {result}")
    print(f"err: {err}")


def test_dataloader():
    loader = DataLoader(
        ctx_dir=Path(Path(file_dir).parent, "resources/processed_data").as_posix(),
        inference_dir=Path(Path(file_dir).parent, "resources/inferenced").as_posix(),
    )

    pdf_metadata_map = loader.load_pdf_metadata_map()
    assert len(pdf_metadata_map) == 997

    cls_map = loader.load_classification_map()
    assert len(cls_map) > 0

    keywords_map = loader.load_keywords_map()
    assert len(keywords_map) > 0

    nl2sql_map = loader.load_nl2sql_map()
    assert len(nl2sql_map) > 0

    comp_table = loader.load_company_table()
    assert len(comp_table) > 0

    rows = loader.find_company_table_data(
        company="一品红药业股份有限公司", years=[2020]
    )
    assert len(rows) > 0

    txt_lines = loader.load_pdf_pure_text_alltxt(
        "2020-02-20__上海家化联合股份有限公司__600315__上海家化__2019年__年度报告"
    )
    assert len(txt_lines) > 0

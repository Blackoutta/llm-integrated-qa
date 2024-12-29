from ._browser import PreprocessorBrowser
from collections import Counter
from ._table_builder_util import *
import os
import json
import pandas as pd
from tqdm import tqdm

table_names = [
    "basic_info",
    "employee_info",
    "cbs_info",
    "cscf_info",
    "cis_info",
    "dev_info",
]


class PreprocessorTableBuilder(object):
    def __init__(self, ctx_dir: str, doc_dir_name: str = "docs"):
        self.ctx_dir = ctx_dir
        self.doc_dir = os.path.join(ctx_dir, doc_dir_name)
        self.browser = PreprocessorBrowser(doc_dir=self.doc_dir)

    def _gen_table_key_counts(self, persist=False) -> Dict:
        table_df = self.browser.get_tables_as_df()
        page_df = self.browser.get_pure_content_as_df()

        all_keys = []
        for index, row in table_df.iterrows():
            table_values = list(row[table_names])
            key, year = row[["metadata.key", "metadata.year"]]
            searched_page_df = page_df[page_df["metadata.key"] == key].get("text_lines")
            pages = searched_page_df.iat[0] if len(searched_page_df) > 0 else []

            for i, table_name in enumerate(table_names):
                r = table_to_tuples(key, year, table_name, table_values[i], pages)
                if len(r) < 1:
                    continue
                row_names = [t[2] for t in r]
                all_keys.extend(row_names)

        counter = Counter(all_keys)
        if persist:
            with open(
                os.path.join(self.ctx_dir, "key_count.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(counter, f, ensure_ascii=False, indent=4)
        return counter

    def build_table(self, min_ratio=0.05, persist=False) -> pd.DataFrame:
        page_df = self.browser.get_pure_content_as_df()
        table_df = self.browser.get_tables_as_df()
        key_counts = self._gen_table_key_counts()

        max_count = max(key_counts.values())
        sorted_key_counts = sorted(key_counts.items(), key=lambda x: x[1], reverse=True)
        valid_keys = [
            key for key, count in sorted_key_counts if count > min_ratio * max_count
        ]
        columns = ["公司全称", "年份"] + valid_keys
        df_dict = {}
        for col in columns:
            df_dict[col] = ["NULL"] * len(table_df)

        for index, row in tqdm(
            table_df.iterrows(), total=len(table_df), desc="building table"
        ):
            table_values = list(row[table_names])
            key, year, company = row[
                ["metadata.key", "metadata.year", "metadata.company"]
            ]
            year = year.replace("年", "")
            searched_page_df = page_df[page_df["metadata.key"] == key].get("text_lines")
            pages = searched_page_df.iat[0] if len(searched_page_df) > 0 else []

            df_dict["公司全称"][index] = company
            df_dict["年份"][index] = year

            rename_map = {}
            for i, table_name in enumerate(table_names):
                """
                table_to_tuples() returns:
                [
                (table name, year, col name, value),
                (table name, year, col name, value),
                ]
                """
                r = table_to_tuples(key, year, table_name, table_values[i], pages)
                if len(r) < 1:
                    continue
                for t in r:
                    table_name = t[0]
                    col_name = t[2]
                    col_value = t[3]
                    if col_name not in valid_keys:
                        continue
                    col_value = (
                        col_value.replace("人", "").replace("元", "").replace(" ", "")
                    )
                    df_dict[col_name][index] = col_value
                    full_col_name = f"{table_name}.{col_name}"
                    rename_map[col_name] = full_col_name

        final_df = pd.DataFrame(df_dict)
        final_df.sort_values(by=["公司全称", "年份"], inplace=True)
        final_df.rename(columns=rename_map, inplace=True)
        if persist:
            final_df.to_csv(
                os.path.join(self.ctx_dir, "CompanyTable.csv"),
                sep="\t",
                index=False,
                encoding="utf-8",
            )
        return final_df


"""
Tests
"""


def test_gen_table_key_counts():
    builder = PreprocessorTableBuilder(
        ctx_dir=os.path.join(os.path.dirname(__file__), "test_temp")
    )
    counter = builder._gen_table_key_counts()
    print(counter)


def test_build_table():
    builder = PreprocessorTableBuilder(
        ctx_dir=os.path.join(os.path.dirname(__file__), "test_temp")
    )
    df = builder.build_table(persist=True)
    print(df)

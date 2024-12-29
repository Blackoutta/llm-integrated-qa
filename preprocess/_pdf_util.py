import os
import json
import shutil
import tempfile
import pdfplumber
import camelot
from loguru import logger


class PdfExtractor(object):

    def __init__(self, path, xpdf_path="") -> None:
        self.path = path
        self.xpdf_path = xpdf_path

    def extract_pure_content_and_save(self, save_path, use_xpdf=True):
        try:
            pdf = pdfplumber.open(self.path)
        except Exception as e:
            logger.error(f"error opening {self.path} with pdfplumber: {e}")
            return

        if not use_xpdf:
            with open(save_path, "w", encoding="utf-8") as f:
                for page in pdf.pages:
                    text = page.extract_text()
                    line = {"page": page.page_number, "text": text}
                    f.write(json.dumps(line, ensure_ascii=True) + "\n")
        else:
            os.chdir(self.xpdf_path)
            cmd = './pdftotext -table -enc UTF-8 "{}" "{}"'.format(self.path, save_path)
            try:
                os.system(cmd)
            except Exception as e:
                logger.error(f"error running {cmd}: {e}")
                return
            with open(save_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                pages = "\n".join(lines).split("\x0c")

            if len(pdf.pages) != len(pages) - 1:
                logger.error(
                    "{} {} does not match for {}".format(
                        len(pdf.pages), len(pages), self.path
                    )
                )
            with open(save_path, "w", encoding="utf-8") as f:
                for idx, page in enumerate(pages):
                    lines = page.split("\n")
                    lines = [line for line in lines if len(line.strip()) > 0]
                    page_block = {"page": idx + 1, "text": "\n".join(lines)}
                    f.write(json.dumps(page_block, ensure_ascii=False) + "\n")
        pdf.close()

    def extract_table_of_pages(self, page_ids: list):
        """
        this method is slow
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            if not self.path.endswith(".pdf"):
                temp_path = os.path.join(
                    temp_dir, "{}.pdf".format(os.path.basename(self.path))
                )
                shutil.copy(self.path, temp_path)
            else:
                temp_path = self.path
            try:
                tables = camelot.read_pdf(
                    temp_path,
                    strip_text="\n",
                    pages=",".join(map(str, page_ids)),
                    line_tol=6,
                    line_scale=60,
                )
            except IndexError:
                logger.error(f"error reading {temp_path} with camelot: {e}")
                tables = []

            # check chaos tables
            num_chaos = 0
            for table in tables:
                for _, row in table.df.iterrows():
                    for v in row.values:
                        point_num = list(v).count(".")
                        num_chaos = max(point_num, num_chaos)
            # print(num_chaos, '--')

            if len(tables) == 0 or num_chaos > 5:
                tables = camelot.read_pdf(
                    temp_path,
                    pages=",".join(map(str, page_ids)),
                    flavor="stream",
                    edge_tol=100,
                )

        return tables

    def extract_table_of_pages_pdfplumber(self, page_ids: list):
        pdf = pdfplumber.open(self.path)

        tables = []
        for i, page in enumerate(pdf.pages):
            if page.page_number not in page_ids:
                continue

            page = page.filter(self.keep_visible_lines)

            edges = self.curves_to_edges(page.curves + page.edges)
            if len(edges) > 0:
                table_settings = {
                    "vertical_strategy": "explicit",
                    "horizontal_strategy": "explicit",
                    "explicit_vertical_lines": self.curves_to_edges(
                        page.curves + page.edges
                    ),
                    "explicit_horizontal_lines": self.curves_to_edges(
                        page.curves + page.edges
                    ),
                    "intersection_y_tolerance": 3,
                    "snap_tolerance": 3,
                }
            else:
                table_settings = {
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "snap_tolerance": 3,
                }

            # Get the bounding boxes of the tables on the page.
            plumber_tables = page.find_tables(table_settings=table_settings)
            tables.extend([self.get_text(t) for t in plumber_tables])

        # for table in tables:
        #     print(table)

        return tables

    @staticmethod
    def keep_visible_lines(obj):
        """
        If the object is a ``rect`` type, keep it only if the lines are visible.

        A visible line is the one having ``non_stroking_color`` not null.
        """
        if obj["object_type"] == "rect":
            if obj["non_stroking_color"] is None:
                return False
            if obj["width"] < 1 and obj["height"] < 1:
                return False
            # return obj['width'] >= 1 and obj['height'] >= 1 and obj['non_stroking_color'] is not None
        if obj["object_type"] == "char":
            return (
                obj["stroking_color"] is not None
                and obj["non_stroking_color"] is not None
            )
        return True

    @staticmethod
    def curves_to_edges(cs):
        """See https://github.com/jsvine/pdfplumber/issues/127"""
        edges = []
        for c in cs:
            edges += pdfplumber.utils.rect_to_edges(c)
        return edges

    @staticmethod
    def not_within_bboxes(obj, bboxes):
        """Check if the object is in any of the table's bbox."""

        def obj_in_bbox(_bbox):
            """See https://github.com/jsvine/pdfplumber/blob/stable/pdfplumber/table.py#L404"""
            v_mid = (obj["top"] + obj["bottom"]) / 2
            h_mid = (obj["x0"] + obj["x1"]) / 2
            x0, top, x1, bottom = _bbox
            return (
                (h_mid >= x0) and (h_mid < x1) and (v_mid >= top) and (v_mid < bottom)
            )

        return not any(obj_in_bbox(__bbox) for __bbox in bboxes)

    @staticmethod
    def get_top(obj):
        if isinstance(obj, pdfplumber.table.Table):
            return obj.bbox[1]
        if isinstance(obj, dict):
            return obj["top"]

    @staticmethod
    def get_text(obj):
        if isinstance(obj, pdfplumber.table.Table):
            table_text = obj.extract()
            table_text = [
                [t if t is not None else "NULL" for t in row] for row in table_text
            ]
            table_text = [
                [t.replace("\n", "").replace(" ", "") for t in row]
                for row in table_text
            ]
            table_text = [[t if t != "" else "NULL" for t in row] for row in table_text]
            text = "\n"
            if len(table_text) == 0:
                return text
            num_cols = len(table_text[0])
            seps = ["---" for _ in range(num_cols)]
            if len(table_text) > 1:
                table_text.insert(1, seps)
            for row in table_text:
                text += "| {} |\n".format(" | ".join(row))
            text += "\n"
            return text
        if isinstance(obj, dict):
            text = obj["text"].replace(" ", "").replace("\n", "")
            if len(text) == 0:
                return ""
            else:
                return text + "\n"

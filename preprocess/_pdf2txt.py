import glob
import pdfplumber
import re
from collections import defaultdict
import json
import os
from tqdm import tqdm
from loguru import logger


class PDFProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.pdf = pdfplumber.open(filepath)
        self.all_text = defaultdict(dict)
        self.allrow = 0
        self.last_num = 0

    def check_lines(self, page, top, buttom):
        lines = page.extract_words()[::]
        text = ""
        last_top = 0
        last_check = 0
        for l in range(len(lines)):
            each_line = lines[l]
            check_re = "(?:。|；|单位：元|单位：万元|币种：人民币|\d|报告(?:全文)?(?:（修订版）|（修订稿）|（更正后）)?)$"
            if top == "" and buttom == "":
                if abs(last_top - each_line["top"]) <= 2:
                    text = text + each_line["text"]
                elif (
                    last_check > 0
                    and (page.height * 0.9 - each_line["top"]) > 0
                    and not re.search(check_re, text)
                ):

                    text = text + each_line["text"]
                else:
                    text = text + "\n" + each_line["text"]
            elif top == "":
                if each_line["top"] > buttom:
                    if abs(last_top - each_line["top"]) <= 2:
                        text = text + each_line["text"]
                    elif (
                        last_check > 0
                        and (page.height * 0.85 - each_line["top"]) > 0
                        and not re.search(check_re, text)
                    ):
                        text = text + each_line["text"]
                    else:
                        text = text + "\n" + each_line["text"]
            else:
                if each_line["top"] < top and each_line["top"] > buttom:
                    if abs(last_top - each_line["top"]) <= 2:
                        text = text + each_line["text"]
                    elif (
                        last_check > 0
                        and (page.height * 0.85 - each_line["top"]) > 0
                        and not re.search(check_re, text)
                    ):
                        text = text + each_line["text"]
                    else:
                        text = text + "\n" + each_line["text"]
            last_top = each_line["top"]
            last_check = each_line["x1"] - page.width * 0.85

        return text

    def drop_empty_cols(self, data):
        # 删除所有列为空数据的列
        transposed_data = list(map(list, zip(*data)))
        filtered_data = [
            col for col in transposed_data if not all(cell == "" for cell in col)
        ]
        result = list(map(list, zip(*filtered_data)))
        return result

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

    def extract_text_and_tables(self, page):
        buttom = 0
        page = page.filter(self.keep_visible_lines)
        tables = page.find_tables()
        if len(tables) >= 1:
            count = len(tables)
            for table in tables:
                if table.bbox[3] < buttom:
                    pass
                else:
                    count -= 1
                    top = table.bbox[1]
                    text = self.check_lines(page, top, buttom)
                    text_list = text.split("\n")
                    for _t in range(len(text_list)):
                        self.all_text[self.allrow] = {
                            "page": page.page_number,
                            "allrow": self.allrow,
                            "type": "text",
                            "inside": text_list[_t],
                        }
                        self.allrow += 1

                    buttom = table.bbox[3]
                    new_table = table.extract()
                    r_count = 0
                    for r in range(len(new_table)):
                        row = new_table[r]
                        if row[0] is None:
                            r_count += 1
                            for c in range(len(row)):
                                if row[c] is not None and row[c] not in ["", " "]:
                                    if new_table[r - r_count][c] is None:
                                        new_table[r - r_count][c] = row[c]
                                    else:
                                        new_table[r - r_count][c] += row[c]
                                    new_table[r][c] = None
                        else:
                            r_count = 0

                    end_table = []
                    for row in new_table:
                        if row[0] != None:
                            cell_list = []
                            cell_check = False
                            for cell in row:
                                if cell != None:
                                    cell = cell.replace("\n", "")
                                else:
                                    cell = ""
                                if cell != "":
                                    cell_check = True
                                cell_list.append(cell)
                            if cell_check == True:
                                end_table.append(cell_list)
                    end_table = self.drop_empty_cols(end_table)

                    for row in end_table:
                        self.all_text[self.allrow] = {
                            "page": page.page_number,
                            "allrow": self.allrow,
                            "type": "excel",
                            "inside": str(row),
                        }
                        self.allrow += 1

                    if count == 0:
                        text = self.check_lines(page, "", buttom)
                        text_list = text.split("\n")
                        for _t in range(len(text_list)):
                            self.all_text[self.allrow] = {
                                "page": page.page_number,
                                "allrow": self.allrow,
                                "type": "text",
                                "inside": text_list[_t],
                            }
                            self.allrow += 1

        else:
            text = self.check_lines(page, "", "")
            text_list = text.split("\n")
            for _t in range(len(text_list)):
                self.all_text[self.allrow] = {
                    "page": page.page_number,
                    "allrow": self.allrow,
                    "type": "text",
                    "inside": text_list[_t],
                }
                self.allrow += 1

        first_re = "[^计](?:报告(?:全文)?(?:（修订版）|（修订稿）|（更正后）)?)$"
        end_re = "^(?:\d|\\|\/|第|共|页|-|_| ){1,}"
        if self.last_num == 0:
            try:
                first_text = str(self.all_text[1]["inside"])
                end_text = str(self.all_text[len(self.all_text) - 1]["inside"])
                if re.search(first_re, first_text) and not "[" in end_text:
                    self.all_text[1]["type"] = "页眉"
                    if re.search(end_re, end_text) and not "[" in end_text:
                        self.all_text[len(self.all_text) - 1]["type"] = "页脚"
            except Exception as e:
                logger.error(f"Error processing page {page.page_number}: {e}")
        else:
            try:
                first_text = str(self.all_text[self.last_num + 2]["inside"])
                end_text = str(self.all_text[len(self.all_text) - 1]["inside"])
                if re.search(first_re, first_text) and "[" not in end_text:
                    self.all_text[self.last_num + 2]["type"] = "页眉"
                if re.search(end_re, end_text) and "[" not in end_text:
                    self.all_text[len(self.all_text) - 1]["type"] = "页脚"
            except Exception as e:
                logger.error(f"Error processing page {page.page_number}: {e}")

        self.last_num = len(self.all_text) - 1

    def process_pdf(self):
        for i in range(len(self.pdf.pages)):
            self.extract_text_and_tables(self.pdf.pages[i])

    def save_all_text(self, path):
        with open(path, "w", encoding="utf-8") as file:
            for key in self.all_text.keys():
                file.write(json.dumps(self.all_text[key], ensure_ascii=False) + "\n")


def process_all_pdfs_in_folder(target_pdf_paths, ctx_dir, skip_if_exists=True):
    file_paths = sorted(target_pdf_paths, reverse=True)
    all_txts_dir = os.path.join(ctx_dir, "alltxts")
    if not os.path.exists(all_txts_dir):
        os.makedirs(all_txts_dir)

    for file_path in file_paths:
        try:
            target_file_name = file_path.split("/")[-1].replace(".pdf", ".txt")
            save_path = os.path.join(all_txts_dir, target_file_name)
            if skip_if_exists and os.path.exists(save_path):
                continue
            processor = PDFProcessor(file_path)
            processor.process_pdf()
            processor.save_all_text(save_path)
        except Exception as e:
            logger.error(f"Error processing pdf file {file_path}: {e}")

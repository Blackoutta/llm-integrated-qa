from preprocess import Preprocessor
import os
from loguru import logger
from typing import List
from dotenv import load_dotenv


def parse_target_pdf_paths(metrics_dir: str, pdf_dir: str) -> List[str]:
    with open(os.path.join(metrics_dir, "target_pdfs.txt"), "r", encoding="utf-8") as f:
        lines = f.readlines()
        paths = [os.path.join(pdf_dir, line.strip()) for line in lines]
    for path in paths:
        assert os.path.exists(path), f"{path} does not exist"
    return paths


def main():
    """
    运行前需设置环境变量
    - PDF_DIR: pdf文件所在的目录
    """

    load_dotenv()
    pdf_dir = os.environ.get("PDF_DIR")

    ctx_dir = os.path.abspath(os.path.join("resources", "processed_data"))
    xpdf_dir = os.path.abspath(
        os.path.join("preprocess", "xpdf-tools-linux-4.05", "bin64")
    )
    metrics_dir = os.path.abspath(os.path.join("resources", "metrics"))
    num_processors = os.cpu_count() // 2 + 1

    target_pdf_paths = parse_target_pdf_paths(metrics_dir, pdf_dir)
    logger.info(f"{len(target_pdf_paths)} pdfs used as targets")
    logger.info(f"target pdf path example: {target_pdf_paths[0]}")

    preprocessor = Preprocessor(ctx_dir=ctx_dir, doc_dir_name="docs")

    # pdf2txt
    preprocessor.parse_all_pdf_2_txt(
        target_pdf_paths, num_processors=num_processors, skip_existing=True
    )

    # metadata
    logger.info("generating pdf metadata...")

    preprocessor.gen_pdf_metadata(
        target_pdf_paths=target_pdf_paths,
        persist=True,
    )

    # text and tables
    logger.info("extracting pdf text...")
    preprocessor.extract_pdf_text(xpdf_path=xpdf_dir, num_processors=num_processors)

    logger.info("extracting pdf tables...")
    preprocessor.extract_pdf_tables(num_processors=num_processors)

    # check
    logger.info("checking extractions...")
    err_cnt = preprocessor.check_extractions(
        persist_err_report=True, delete_malformed_pure_content=True
    )
    logger.info(f"{err_cnt} pdfs have extraction errors")

    # build final table
    logger.info("building tables...")
    preprocessor.build_table(persist=True)


if __name__ == "__main__":
    main()

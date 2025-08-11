import os
from PyPDF2 import PdfReader, PdfWriter


def split_pdf_into_pages(pdf_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    reader = PdfReader(pdf_path)
    pages = []
    for i in range(len(reader.pages)):
        writer = PdfWriter()
        writer.add_page(reader.pages[i])
        page_path = os.path.join(output_dir, f"page_{i + 1}.pdf")
        with open(page_path, "wb") as page_file:
            writer.write(page_file)
        pages.append(page_path)
    return pages

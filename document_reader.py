import pymupdf


def extract_pages(source):
    """
    Extract text from every PDF page.
    Page numbers are stored with the text.
    """

    if isinstance(source, (bytes, bytearray)):
        pdf = pymupdf.open(
            stream=source,
            filetype="pdf"
        )
    else:
        pdf = pymupdf.open(source)

    pages = []

    for page_number, page in enumerate(pdf, start=1):

        text = page.get_text("text").strip()

        if text:
            pages.append({
                "page": page_number,
                "text": text
            })

    pdf.close()

    return pages
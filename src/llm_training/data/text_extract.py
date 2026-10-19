from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.parse import encoding
from resiliparse.extract import html2text


def html_to_text(html: bytes) -> str:
    # detect encoding with resiliparse
    enc = encoding.detect_encoding(html)
    unicode = html.decode(enc)

    text = html2text.extract_plain_text(unicode)
    return text


if __name__ == "__main__":
    # test on warc in local-shared-data/CC
    with open("local-shared-data/CC/example.warc.gz", "rb") as f:
        for record in ArchiveIterator(f, record_types=WarcRecordType.request | WarcRecordType.response):
            body = record.reader.read(1024 * 10)
            text = html_to_text(body)
            print(text)

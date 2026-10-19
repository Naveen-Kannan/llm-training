from pathlib import Path
import os
from fastwarc.warc import ArchiveIterator, WarcRecordType
from llm_training.data.common import get_shared_assets_path
from llm_training.data.text_extract import html_to_text
from llm_training.data.gopher_filters import gopher_filters
from llm_training.data.identify_lang import language_identify
from llm_training.data.identify_harmful import identify_nsfw, identify_toxic_speech
from llm_training.data.mask_pii import email_mask, phone_mask, ip_mask
import fasttext

eng_classifier = fasttext.load_model(str(get_shared_assets_path() / "classifiers/lid.176.bin"))
nsfw_classifier = fasttext.load_model(
    str(get_shared_assets_path() / "classifiers/dolma_fasttext_nsfw_jigsaw_model.bin")
)
toxic_classifier = fasttext.load_model(
    str(get_shared_assets_path() / "classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin")
)


def filter(text: str) -> str | bool:
    lang, lang_score = language_identify(text, eng_classifier)
    if lang != "en" or lang_score < 0.7:
        return False
    nsfw_flag, nsfw_score = identify_nsfw(text, nsfw_classifier)
    if nsfw_flag == "nsfw" and nsfw_score > 0.7:
        return False
    toxic_flag, toxic_score = identify_toxic_speech(text, toxic_classifier)
    if toxic_flag == "toxic" and toxic_score > 0.7:
        return False

    if not gopher_filters(text):
        return False

    # masks
    text = ip_mask(phone_mask(email_mask(text)[0])[0])[0]
    # make one line
    text = " ".join(text.split())

    return text


def filter_and_write(f_out, f_in, label, max=float("inf"), is_wet=False):
    count = 0
    for rec in ArchiveIterator(f_in, record_types=WarcRecordType.response | WarcRecordType.conversion):
        if is_wet:
            text = rec.reader.read().decode("utf-8", errors="replace")
        else:
            try:
                text = html_to_text(rec.reader.read())
            except Exception:
                continue
        try:
            filtered = filter(text)
        except Exception:
            continue
        if filtered is False:
            continue
        example_line = f"__label__{label} {filtered}"
        f_out.write(example_line + "\n")
        count += 1
        if count >= max:
            break
    return count


#### need labels on each line for fasttext
if __name__ == "__main__":
    local = get_shared_assets_path()
    data = Path("data")

    # p+
    n_pos = 0
    # n_pos = 10082
    with open(data / "positives.txt", "w") as f_out:
        for warc_file in os.listdir(data / "positive_warcs"):
            with open(data / "positive_warcs" / warc_file, "rb") as f_in:
                n_pos += filter_and_write(f_out, f_in, "wiki")
    print(f"+: {n_pos}")

    n_neg = 0
    with open(data / "negatives.txt", "w") as f_out:
        # theres only one wet file
        for wet_file in os.listdir(local / "english-wet-data"):
            with open(local / "english-wet-data" / wet_file, "rb") as f_in:
                n_neg += filter_and_write(f_out, f_in, "cc", max=n_pos, is_wet=True)
    print(f"-: {n_neg}")

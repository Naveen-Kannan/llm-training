import fasttext
from llm_training.data.text_extract import html_to_text
from fastwarc.warc import ArchiveIterator, WarcRecordType


def identify_nsfw(text: str, model=None) -> str:
    text = text.replace("\n", " ")
    if model is None:
        model = fasttext.load_model("local-shared-data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin")
    prediction = model.predict(text)
    # print(prediction)
    label = prediction[0][0][9:]
    score = prediction[1][0]
    return label, score


def identify_toxic_speech(text: str, model=None) -> str:
    text = text.replace("\n", " ")
    if model is None:
        model = fasttext.load_model("local-shared-data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin")
    prediction = model.predict(text)
    # print(prediction)
    label = prediction[0][0][9:]
    score = prediction[1][0]
    return label, score


if __name__ == "__main__":
    pass
    # print(identify_harmful("This is a test"))
    # predictions = []
    # erors = []
    # with open('local-shared-data/CC/example.warc.gz', 'rb') as f:
    #     for record in ArchiveIterator(f, record_types=WarcRecordType.request | WarcRecordType.response):
    #         body = record.reader.read()
    #         try :
    #             text = html_to_text(body)
    #             label, score = language_identify(text)
    #             print(label, score)
    #             predictions.append((label, score))
    #             print(text)
    #         except Exception as e:
    #             erors.append(e)

    # en = 0
    # for prediction in predictions:
    #     if prediction[0] == 'en':
    #         en += 1
    # print(f"English: {en} / {len(predictions)}")
    # print(f"Errors")
    # print(erors)

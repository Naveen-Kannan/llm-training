import os
import random
from llm_training.data.mask_pii import ip_mask, phone_mask, email_mask
import fasttext


def classify_quality(text: str, model=None) -> bool:
    if model is None:
        model = fasttext.load_model("data/quality_classifier.bin")

    # run our prepocessing
    text = ip_mask(phone_mask(email_mask(text)[0])[0])[0]
    text = " ".join(text.split())
    prediction = model.predict(text)
    label = prediction[0][0][9:]
    score = prediction[1][0]
    return label, score


if __name__ == "__main__":
    # train our simple classifier.
    # not this is done on filtered data.
    os.makedirs("data", exist_ok=True)

    with open("data/temp.txt", "w") as f:
        with open("data/positives.txt") as p:
            for line in p:
                f.write(line)
        with open("data/negatives.txt") as n:
            for line in n:
                f.write(line)

    with open("data/temp.txt") as f:
        lines = f.readlines()

    random.shuffle(lines)
    split = int(0.95 * len(lines))
    train = lines[:split]
    valid = lines[split:]

    os.remove("data/temp.txt")

    with open("data/quality_train.txt", "w") as f:
        for line in train:
            f.write(line)
    with open("data/quality_valid.txt", "w") as f:
        for line in valid:
            f.write(line)

    model = fasttext.train_supervised("data/quality_train.txt", epoch=20, lr=0.5)
    print("acc:", model.test("data/quality_valid.txt"))
    model.save_model("data/quality_classifier.bin")

import nltk

nltk.download("punkt_tab", quiet=True)  # shouldnt this be downloaded?


def gopher_filters(text):
    words = nltk.word_tokenize(text)
    if len(words) < 50 or len(words) > 100000:
        return False

    alpha = 0
    length_sum = 0
    for word in words:
        for letter in word:
            if letter.isalpha():
                alpha += 1
                break
        length_sum += len(word)
    if length_sum / len(words) < 3 or length_sum / len(words) > 10:
        return False
    if alpha / len(words) < 0.8:
        return False

    e_count = 0
    lines = text.split("\n")
    for line in lines:
        if line[-3:] == "...":
            e_count += 1
    if e_count / len(lines) > 0.3:
        return False

    return True

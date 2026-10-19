import regex as re


def email_mask(text: str):
    mask = r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"
    # need to return new str replaced and number
    return re.subn(mask, "|||EMAIL_ADDRESS|||", text)


def phone_mask(text: str):
    plain = r"\b[0-9]{10}\b"
    dashed = r"\b[0-9]{3}-[0-9]{3}-[0-9]{4}\b"
    parens = r"\([0-9]{3}\)[0-9]{3}-[0-9]{4}\b"
    parens_dash = r"\([0-9]{3}\)-[0-9]{3}-[0-9]{4}\b"
    parens_space = r"\([0-9]{3}\) [0-9]{3} [0-9]{4}\b"

    mask = rf"(?:{plain}|{dashed}|{parens}|{parens_dash}|{parens_space})"

    return re.subn(mask, "|||PHONE_NUMBER|||", text)


def ip_mask(text: str):
    # less 255
    num = r"(?:[0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])"
    mask = rf"\b{num}\.{num}\.{num}\.{num}\b"
    return re.subn(mask, "|||IP_ADDRESS|||", text)

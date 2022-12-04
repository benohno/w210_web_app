from readability import Readability


def readability_score(text):
    """

    :param text: privacy policy raw text
    :returns: Flesch–Kincaid readability score

    """
    r = Readability(text)

    return round(r.flesch_kincaid().score)


def word_count(text):
    """

    :param text: privacy policy raw text
    :returns: total number of words in text

    """
    return (len(text.strip().split(" ")))

import string
import math
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SpanishStemmer


def bag_of_words(documents: list) -> (set, list):
    """
    Transform a text in a 'bag of words' with words occurrences.
    The first component of the tuple is the set of words, and the second
    component is the occurrences (words, occurrences).
    """

    # Join all documents in a line and transform to lowercase
    text = (' '.join(documents)).lower()

    # Load spanish stop words
    stop_words = set(stopwords.words('spanish'))

    # Remove string punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # Tokenize text removing duplicates and stop words
    words = set([word for word in word_tokenize(text, 'spanish') if word not in stop_words])

    # Get occurrences for each word in words
    occurrences = [text.count(word) for word in words]

    return words, occurrences


def similarity_words(v: list, w: list) -> float:
    """
    Get the similarity between words vectors.
    """

    if len(v) != len(w):
        raise ValueError("Both vectors must be have same length")

    vw = sum([a * b for a, b in zip(v, w)])
    v_sqrt = math.sqrt(sum([math.pow(a, 2) for a in v]))
    w_sqrt = math.sqrt(sum([math.pow(a, 2) for a in w]))

    return vw / (v_sqrt * w_sqrt)


def tf(word: str, document: str) -> int:
    """
    Get occurrences of word in document (tf).
    """
    return document.lower().count(word.lower())


def df(word: str, documents: list) -> int:
    """
    Get numbers of documents where appears the word (df).
    """
    return sum([1 for document in documents if tf(word, document) > 0])


def idf(word: str, documents: list) -> float:
    """
    Get inverse documentary frequency of the word.
    """
    n = len(documents)
    res_df = df(word, documents)
    return math.log(n / res_df, 10)


def tfidf(word: str, document: str, documents: list) -> float:
    """
    Get weight of the word in a specific document.
    """
    return tf(word, document) * idf(word, documents)


def stemming(word: str) -> str:
    """
    Simply function to transform a word to a stem of that word (In Spanish)
    :param word: str
    :return stem of word: str
    """
    return SpanishStemmer().stem(word.lower())


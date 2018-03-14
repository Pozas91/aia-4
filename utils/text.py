import string
import math
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

"""
Transform a text in a 'bag of words' with words occurrences.
The first component of the tuple is the set of words, and the second
component is the occurrences (words, occurrences).
"""


def bag_of_words(text: str) -> (set, list):
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


"""
Get the similarity between words vectors.
"""


def similarity_words(v: list, w: list) -> float:
    if len(v) != len(w):
        raise ValueError("Both vectors must be have same length")

    vw = sum([a * b for a, b in zip(v, w)])
    v_sqrt = math.sqrt(sum([math.pow(a, 2) for a in v]))
    w_sqrt = math.sqrt(sum([math.pow(a, 2) for a in w]))

    return vw / (v_sqrt * w_sqrt)


_, occurrences = bag_of_words("Juan quiere comprar un coche. Ana no quiere comprar ningún coche.")

print(similarity_words(occurrences, occurrences))

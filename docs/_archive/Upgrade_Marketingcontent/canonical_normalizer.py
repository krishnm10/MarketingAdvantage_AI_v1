# canonical_normalizer.py

import re
import unicodedata
from nltk.stem import PorterStemmer

stop_words = {"and", "&", "the", "of", "products", "product", "items"}

stemmer = PorterStemmer()


def canonical_normalize(raw: str) -> str:
    # Normalize unicode
    n = unicodedata.normalize("NFKD", raw)

    # Lowercase
    n = n.lower()

    # Replace symbols
    n = n.replace("&", " and ")

    # Remove special chars
    n = re.sub(r"[^a-z0-9\s-]", "", n)

    # Tokenize
    tokens = [t for t in n.split() if t not in stop_words]

    # Stem
    stemmed = [stemmer.stem(t) for t in tokens]

    # Join with hyphens
    canonical = "-".join(stemmed)

    # Remove double hyphens
    canonical = re.sub("-+", "-", canonical)

    return canonical.strip("-")

from typing import Callable
from typing import List

import spacy


def sentencizer(name: str = 'en_core_web_sm') -> Callable[[str], List[str]]:
    """ Return a function which splits a document text into sentences.
    Please note that you need to download a model:
        $ python -m spacy download en_core_web_sm
    Here, we download a best-matching default model. Take a look at
    the documentation: https://spacy.io/models/en for more details. """
    nlp = spacy.load(name)

    def wrapper(text: str) -> List[str]:
        doc = nlp(text)
        sentences = [str(sent).strip() for sent in doc.sents]
        return sentences

    return wrapper

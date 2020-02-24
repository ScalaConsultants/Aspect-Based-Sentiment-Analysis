from typing import Iterable
from typing import List

import transformers

from .data_types import ClassifierExample
from .data_types import ExtractorExample
from .data_types import MultimodalExample
from .models import BertABSClassifier
from .pipelines import BertPipeline


def pipeline(name: str = 'absa/bert-rest-0.1'):
    if name is 'absa/bert-rest-0.1':
        model = BertABSClassifier.from_pretrained(name)
        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        return BertPipeline(tokenizer, model)
    raise ValueError('Specified model is not supported')


def load_docs(fname: str) -> Iterable[List[str]]:
    """  """


def load_extractor_examples(fname: str) -> Iterable[ExtractorExample]:
    """  """


def load_classifier_examples(fname: str) -> Iterable[ClassifierExample]:
    """  """


def load_multimodal_examples(fname: str) -> Iterable[MultimodalExample]:
    """  """

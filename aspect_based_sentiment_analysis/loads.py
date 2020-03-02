import os
from typing import Iterable
from typing import List

import transformers
from google.cloud.exceptions import NotFound

from . import utils
from .data_types import ClassifierExample
from .data_types import ExtractorExample
from .data_types import MultimodalExample
from .models import BertABSClassifier
from .pipelines import BertPipeline


def pipeline(name: str = 'absa/bert-rest-0.1'):
    """ Model weights are stored in the HaggingFace AWS S3. """
    if name is 'absa/bert-rest-0.1':
        model = BertABSClassifier.from_pretrained(name)
        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        return BertPipeline(tokenizer, model)
    raise ValueError('Specified model is not supported')


def load_docs(fname: str) -> Iterable[List[str]]:
    """  """


def load_extractor_examples(fname: str) -> Iterable[ExtractorExample]:
    """  """


def load_classifier_examples(
        dataset: str = 'semeval',
        domain: str = 'laptops',
        test: bool = False
) -> List[ClassifierExample]:
    """ Download a dataset from the bucket if it is needed. """
    try:
        split = 'train' if not test else 'test'
        name = f'classifier-{dataset}-{domain}-{split}.bin'
        local_path = utils.file_from_bucket(name)
        examples = utils.load(local_path)
        return examples
    except NotFound:
        local_path = f'{os.path.dirname(__file__)}/downloads/{name}'
        if os.path.isfile(local_path):
            os.remove(local_path)
        raise ValueError('Dataset not found. Please check a documentation.')


def load_multimodal_examples(fname: str) -> Iterable[MultimodalExample]:
    """  """


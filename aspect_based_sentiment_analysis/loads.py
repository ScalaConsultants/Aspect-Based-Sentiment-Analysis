import os
import logging
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
logger = logging.getLogger('absa.pipeline')


def pipeline(name: str):
    """ Model weights are stored in the HaggingFace AWS S3. """
    try:
        model = BertABSClassifier.from_pretrained(name)
        tokenizer = transformers.BertTokenizer.from_pretrained(name)
        return BertPipeline(tokenizer, model)

    except EnvironmentError as error:
        text = 'Model or Tokenizer not found. Please check a documentation.'
        logger.error(text)
        raise error


def load_docs(fname: str) -> Iterable[List[str]]:
    """  """


def load_extractor_examples(fname: str) -> Iterable[ExtractorExample]:
    """  """


def load_classifier_examples(
        dataset: str = 'semeval',
        domain: str = 'laptop',
        test: bool = False
) -> List[ClassifierExample]:
    """ Download a dataset from the bucket if it is needed. """
    try:
        split = 'train' if not test else 'test'
        name = f'classifier-{dataset}-{domain}-{split}.bin'
        local_path = utils.file_from_bucket(name)
        examples = utils.load(local_path)
        return examples

    except NotFound as error:
        local_path = f'{os.path.dirname(__file__)}/downloads/{name}'
        if os.path.isfile(local_path):
            os.remove(local_path)
        text = 'Dataset not found. Please check a documentation.'
        logger.error(text)
        raise error


def load_multimodal_examples(fname: str) -> Iterable[MultimodalExample]:
    """  """


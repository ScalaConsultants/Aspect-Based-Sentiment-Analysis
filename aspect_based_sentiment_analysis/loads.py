import os
import logging
from typing import Iterable
from typing import List

import transformers
from google.cloud.exceptions import NotFound

from . import utils
from .models import BertABSClassifier
from .pipelines import Pipeline
from .pipelines import BertPipeline
from .training import ClassifierExample

logger = logging.getLogger('absa.pipeline')
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DOWNLOADS_DIR = os.path.join(ROOT_DIR, 'downloads')


def load(name: str = 'absa/classifier-rest-0.1', **kwargs) -> Pipeline:
    """ Files are stored on the HaggingFace AWS S3. """
    try:
        model = BertABSClassifier.from_pretrained(name, **kwargs)
        tokenizer = transformers.BertTokenizer.from_pretrained(name)
        return BertPipeline(model, tokenizer)

    except EnvironmentError as error:
        text = 'Model or Tokenizer not found. Please check a documentation.'
        logger.error(text)
        raise error


def load_docs(fname: str) -> Iterable[List[str]]:
    """  """
    raise NotImplemented


def load_classifier_examples(
        dataset: str = 'semeval',
        domain: str = 'laptop',
        test: bool = False
) -> List[ClassifierExample]:
    """ Download a dataset from the bucket if it is needed. """
    split = 'train' if not test else 'test'
    name = f'classifier-{dataset}-{domain}-{split}.bin'
    local_path = os.path.join(DOWNLOADS_DIR, name)

    try:
        local_path = utils.file_from_bucket(name)
        examples = utils.load(local_path)
        return examples

    except NotFound as error:
        if os.path.isfile(local_path):
            os.remove(local_path)
        text = 'Dataset not found. Please check a documentation.'
        logger.error(text)
        raise error

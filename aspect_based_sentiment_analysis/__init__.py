# Version of the aspect-based-sentiment-analysis package
__version__ = "1.0.1"

from .alignment import make_aspect_span
from .alignment import make_alignment
from .alignment import merge_input_attentions

from .data_types import AspectSpan
from .data_types import AspectSpanLabeled
from .data_types import AspectDocument
from .data_types import AspectDocumentLabeled
from .data_types import Document
from .data_types import DocumentLabeled
from .data_types import InputBatch
from .data_types import OutputBatch
from .data_types import Sentiment

from .loads import load
from .loads import load_docs
from .loads import load_classifier_examples

from .models import BertABSCConfig
from .models import BertABSClassifier

from .pipelines import Pipeline
from .pipelines import BertPipeline

from . import training
from . import probing
from . import utils

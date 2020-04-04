from typing import Iterable
from typing import Iterator
from typing import List
from typing import Tuple

import numpy as np
import transformers

from .utils import random


def split_document(
        document: Iterable[List[str]],
        target_length: int,
        split_ratios: Iterator[float] = random(float)
) -> Iterable[Tuple[List[str], List[str]]]:
    """ Each document is an iterator which contains encoded sentences,
    the list of string tokens. The method aim is to split the document into
    chunks, where a chunk has to contain the target length of tokens. Each
    chunk is divided into two segments according to the split ratio (with
    the constrain that sentence itself can not be divided). """
    chunk = []
    chunk_length = 0
    for sentences in document:

        chunk.append(sentences)
        chunk_length += len(sentences)

        if len(chunk) < 2 or chunk_length < target_length:
            continue

        split_ratio = next(split_ratios)
        split_point = max(1, int(len(chunk) * split_ratio))
        segment_a = list(*chunk[:split_point])
        segment_b = list(*chunk[split_point:])
        yield segment_a, segment_b

        chunk = []
        chunk_length = 0


def add_random_token_pairs(
        segment_pairs: Iterable[Tuple[List[str], List[str]]],
        random_documents: Iterator[Iterable[List[str]]],
        target_length: int,
        is_next_iter: Iterator[bool] = random(bool)
) -> Iterable[Tuple[List[str], List[str], bool]]:
    """ Mix up segment pairs (built from following sentences) with segments
    from randomly selected documents according with the `is_next_iter` iterator
    (random booleans by default).

    Note 1: Random documents should start from a random sentence, otherwise
    model can easily detect weird features, spurious correlations.

    Note 2: In this implementation, unfortunately, we waste a pair (segment_b,
    random_segment) by design, because it can lean to artifacts, and hard to
    debug problems. For `is_next` set to False, we would have two times more
    likely negative examples in row (if we only add the third yield statement).
    Using the big batch size, it does not have an impact, but we would have to
    remember to increase probability of the `is_next` parameter etc. """
    for segment_a, segment_b in segment_pairs:

        is_next = next(is_next_iter)
        if is_next:
            yield segment_a, segment_b, is_next
            continue

        random_document = next(random_documents)
        length = target_length - len(segment_a)
        random_segment = get_segment(random_document, length)
        yield segment_a, random_segment, is_next


def get_segment(document: Iterable[List[str]], target_length: int) -> List[str]:
    """ Get the segment (the sequence of string tokens) from a document which
    contains encoded sentences. Return the segment if the target length is
    achieved, raise ValueError otherwise. """
    segment = []
    for sentence in document:
        segment.extend(sentence)
        if len(segment) < target_length:
            continue
        return segment
    raise ValueError


def truncate_pair(
        segment_a: List[str],
        segment_b: List[str],
        max_num_tokens: int,
        remove_from_end: Iterator[bool] = random(bool)
) -> Tuple[List[str], List[str]]:
    """ Truncates a pair of sequences to a maximum sequence length. """
    segment_a, segment_b = segment_a.copy(), segment_b.copy()
    while True:
        total_length = len(segment_a) + len(segment_b)
        if total_length <= max_num_tokens:
            return segment_a, segment_b

        segment = segment_a if len(segment_a) > len(segment_b) else segment_b
        # We want to sometimes truncate from the front and sometimes
        # from the back to add more randomness and avoid biases.
        if next(remove_from_end):
            segment.pop()
        else:
            segment.pop(0)


def mask_tokens(inputs: np.ndarray,
                tokenizer: transformers.BertTokenizer,
                mlm_probability: float) -> Tuple[np.ndarray, np.ndarray]:
    """ Prepare masked tokens inputs/targets for masked language modeling: 80%
    MASK, 10% random, 10% original (adapted from the HuggingFace repo). """
    # Our target is a partially masked input, so we copy it at the beginning.
    targets = inputs.copy()

    # We sample few tokens in each sequence for masked-LM training with
    # probability `mlm_probability` (defaults to 0.2).
    P = np.full(targets.shape, mlm_probability)

    # We do not mask special or padding tokens by definition.
    special_tokens_mask = np.array([tokenizer.get_special_tokens_mask \
                                    (token_ids, already_has_special_tokens=True)
                                    for token_ids in targets], dtype=bool)
    padding_mask = targets == tokenizer.pad_token_id
    P[special_tokens_mask | padding_mask] = 0

    # Next, we draw the target indices (binary random numbers from a Bernoulli
    # distribution) which we will mask in the input.
    masked_indices = np.random.binomial(n=1, p=P, size=targets.shape).astype(bool)
    # Additionally, we `turn off` other tokens (which are in the input),
    # and only compute loss over masked tokens. Now, targets are set up.
    targets[~masked_indices] = -100

    # In next few lines, we mask the input. As we mentioned before, 80% of the
    # time, we replace masked input tokens with tokenizer.mask_token([MASK]).
    indices_replaced = np.random.binomial(n=1, p=0.8, size=inputs.shape).astype(bool) \
                       & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with a random word.
    indices_random = np.random.binomial(n=1, p=0.5, size=inputs.shape).astype(bool) \
                     & masked_indices & ~indices_replaced
    random_words = np.random.choice(len(tokenizer), targets.shape)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens
    # unchanged (the model can use it directly for a target prediction,
    # however mixed-up with random tokens, it is not so obvious).
    return inputs, targets

from dataclasses import dataclass
from typing import Iterable
from typing import Iterator
from typing import List

import numpy as np
import tensorflow as tf
import transformers

from ..data_types import LanguageModelTrainBatch
from ..data_types import LanguageModelExample
from .datasets import StreamDataset
from .utils import random
from . import language_model_functions


@dataclass(frozen=True)
class DocumentStore:
    """ The document store is a collection of raw documents. Each document is
    represented by the list of strings, where a string can be a sentence or a
    phrase. This implementation uses purely the memory, without any kind of
    serialization. """
    documents: List[List[str]]
    doc_lengths: List[int]
    doc_cumsum: List[int]
    cumsum_max: int

    def __iter__(self):
        """ Loop over the store. """
        return self.documents

    def __getitem__(self, index: int) -> List[str]:
        """ Get a document by the index. """
        return self.documents[index]

    @classmethod
    def from_iterable(cls, documents: Iterable[List[str]]):
        """ Build the store from an iterable. """
        documents = list(documents)
        doc_lengths = list(map(len, documents))
        doc_cumsum = list(np.cumsum(doc_lengths))
        *_, doc_cumsum_max = doc_cumsum
        return cls(documents, doc_lengths, doc_cumsum, doc_cumsum_max)

    def sample_doc(self, current_idx: int) -> List[str]:
        """ Uses the current iteration counter to ensure we don't sample the
        same doc twice. Documents are sampled proportionally to the number of
        sentences they contain, which means each sentence (rather than each
        document) has an equal chance of being sampled as a false example for
        the NextSentence task. """
        rand_start = self.doc_cumsum[current_idx]
        rand_end = rand_start + self.cumsum_max - self.doc_lengths[current_idx]
        sentence_index = np.random.randint(rand_start, rand_end) % self.cumsum_max
        index = np.searchsorted(self.doc_cumsum, sentence_index, side='right')
        return self.documents[index]


@dataclass(frozen=True)
class LanguageModelDataset(StreamDataset):
    """ The Language Model Dataset generates training batches from the
    document store. Please note that the process of generating samples from
    documents is stochastic (in consequences, most of the logic is detached
    to the `language_model_functions` module). Shuffle data after each epoch
    is not available.

    Along with three well-defined parameters, we also have:
    `max_num_tokens`:
        The hard limit how long the entire sequence to process can be
        (literally: the segment A, the segment B and special tokens)
    `mlm_probability`:
        The probability that a token is masked in the input sequence
        (the language model tries to predict them).
    `short_seq_prob`
        To minimize the mismatch between pre-training and post-training,
        sometimes, with the `short_seq_prob` probability, we want to use
        shorter sequences.
    """
    document_store: DocumentStore
    batch_size: int
    tokenizer: transformers.BertTokenizer
    max_num_tokens: int
    mlm_probability: float = 0.2
    short_seq_prob: float = 0.1

    def examples_generator(self) -> Iterable[LanguageModelExample]:
        """ The generator produces language model examples. It is equivalent
        to the single function from Google BERT's or HuggingFace repo. It can
        be easily adjust to the parallel processing (e.g. using queues)."""
        # Iterate over every document in the document store.
        for doc_index, document in enumerate(self.document_store):

            # Reduce `max_num_token` to target length, and minimize the
            # mismatch between pre-training and fine-tuning.
            target_length = self.target_length()

            # Encode each sentence in a document into the list of string tokens.
            tokenized_sentences = (self.tokenizer.tokenize(sentence)
                                   for sentence in document)

            # We DON'T just concatenate all of the tokens from a document
            # into a long sequence and choose an arbitrary split point
            # because this would make the `NextSentence` prediction task too
            # easy. Instead, we split a document into segment pairs "A" and
            # "B" which are composed of complete sentences.
            segment_pairs = language_model_functions \
                .split_document(tokenized_sentences, target_length)

            # We have to add false examples, because now, segment pairs are
            # built from following sentences. We split some of segment pairs,
            # and compose B segments from random documents. Random documents
            # are sampled proportionally to the number of sentences they
            # contain, which means each sentence (rather than each document)
            # has an equal chance of being sampled as a false example.
            random_segments = self.generate_random_segments(doc_index)
            segment_pairs = language_model_functions \
                .add_random_token_pairs(segment_pairs, random_segments, target_length)

            for segment_a, segment_b, is_next in segment_pairs:

                # Adjust segments lengths, and ensure that
                # they do not exceed the limit.
                segment_a, segment_b = language_model_functions \
                    .truncate_pair(segment_a, segment_b, self.max_num_tokens)

                # Decode tokens to strings, therefore language model examples
                # do not relate to the specific tokenizer.
                text_a = self.tokenizer.convert_tokens_to_string(segment_a)
                text_b = self.tokenizer.convert_tokens_to_string(segment_b)
                example = LanguageModelExample(text_a, text_b, is_next)
                yield example

    def preprocess_batch(
            self, batch_examples: List[LanguageModelExample]
    ) -> LanguageModelTrainBatch:
        """ Convert language model examples to the LanguageModelTrainBatch. """
        pairs = [(e.text_a, e.text_b) for e in batch_examples]
        encoded = self.tokenizer.batch_encode_plus(pairs,
                                                   add_special_tokens=True,
                                                   pad_to_max_length=True,
                                                   return_attention_masks=True,
                                                   return_tensors='tf')
        # We plan to build up tensors after encoding because matrices are not
        # so big to feel the difference (more comfortable to work with
        # numpy's), and we do not use `inputs_ids` anymore, so we waste the
        # memory on GPU. However, the batch padding in the `transformer`
        # implementation is available only if you convert to tensors. We do
        # not want to waste time, so at this stage we decide to leave it in
        # this form.
        input_ids = encoded['input_ids'].numpy()
        # Note that masking is performed on the entire batch at once, so the
        # `mlm_probability` can differ among examples.
        token_ids, target_masked_token_ids = language_model_functions \
            .mask_tokens(input_ids, self.tokenizer, self.mlm_probability)

        token_ids = tf.convert_to_tensor(token_ids)
        target_masked_token_ids = tf.convert_to_tensor(target_masked_token_ids)
        attention_mask = encoded['attention_mask']
        token_type_ids = encoded['token_type_ids']
        target_is_next = tf.convert_to_tensor([e.is_next for e in batch_examples])

        return LanguageModelTrainBatch(
            token_ids,
            attention_mask,
            token_type_ids,
            target_masked_token_ids,
            target_is_next
        )

    def target_length(self) -> int:
        """ We *usually* want to fill up the entire sequence since we are
        padding to `max_seq_length` anyways, so short sequences are generally
        wasted computation. However, we want to use shorter sequences to
        minimize the mismatch between pre-training and fine-tuning. """
        if np.random.random() < self.short_seq_prob:
            return np.random.randint(2, self.max_num_tokens)
        else:
            return self.max_num_tokens

    def generate_random_segments(
            self,
            doc_index: int,
            start_indices: Iterator[float] = random(float)
    ) -> Iterator[Iterable[List[str]]]:
        """ This method builds a generator which produces tokenized sentences
        from randomly selected documents. To build language model examples,
        *sometimes* we mix up segments with segments which come from randomly
        selected documents (take a look at `add_random_token_pairs` function).
        Moreover, to avoid biases, we do not want to start from the document
        beginning, we start from a random sentence. """
        def random_segments() -> List[str]:
            while True:
                document = self.document_store.sample_doc(doc_index)
                start_index = int(round(len(document) * next(start_indices)))
                iterable = (self.tokenizer.tokenize(sentence)
                            for sentence in document[start_index:])
                yield iterable
        return random_segments()

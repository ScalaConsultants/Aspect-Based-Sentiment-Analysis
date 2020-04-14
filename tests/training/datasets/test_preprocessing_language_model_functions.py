from mock import MagicMock

import numpy as np
import tensorflow as tf

from aspect_based_sentiment_analysis.training.datasets \
    import language_model_functions


def test_mask_tokens():
    np.random.seed(1)
    tf.random.set_seed(1)

    # Define encoded pairs of segments (batch_size, tokenizer_dim)
    vocabulary_len = 10
    batch_size = 32
    seq_length = 512
    mask_token_id = 1000
    special_token_id = 100
    padding_token_id = -1

    # We fake the input matrix of encoded tokens, wherein are also
    # special and padding tokens.
    inputs = np.random.choice(vocabulary_len, size=[batch_size, seq_length])
    special_tokens = np.random.binomial(n=1, p=0.1, size=inputs.shape).astype(bool)
    inputs[special_tokens] = special_token_id
    padding_tokens = np.random.binomial(n=1, p=0.2, size=inputs.shape).astype(bool) \
                     & ~special_tokens
    inputs[padding_tokens] = padding_token_id

    # We do not test Tokenizer, so we can mock it.
    tokenizer = MagicMock()
    tokenizer.__len__.return_value = vocabulary_len
    tokenizer.pad_token_id = padding_token_id
    tokenizer.mask_token = None
    # We only convert mask token
    tokenizer.convert_tokens_to_ids = lambda _: mask_token_id

    def mock_get_special_tokens_mask(token_ids, already_has_special_tokens):
        return [1 if token == special_token_id else 0 for token in token_ids]

    tokenizer.get_special_tokens_mask = mock_get_special_tokens_mask
    masked_inputs, masked_targets = language_model_functions.mask_tokens(
        inputs, tokenizer, mlm_probability=0.2
    )
    # Our inputs and targets should have the same size as provided
    assert masked_inputs.shape == masked_targets.shape == (32, 512)
    # Special/padding tokens are not changed in the masked_input
    # and they are not in the target.
    assert np.allclose(masked_inputs[special_tokens], special_token_id)
    assert np.allclose(masked_inputs[padding_tokens], padding_token_id)
    assert np.allclose(masked_targets[special_tokens], -100)
    assert np.allclose(masked_targets[padding_tokens], -100)

    # Now, we should check, if our target is masked in the input 80% of time.
    target_to_predict = masked_targets != -100
    mask = masked_inputs[target_to_predict] == mask_token_id
    masked_target_ration = np.sum(mask) / np.sum(target_to_predict)
    assert round(masked_target_ration, 2) == 0.80

import os
import logging
from dataclasses import dataclass
from functools import partial
from typing import Callable

import optuna
import numpy as np
import tensorflow as tf
import transformers
from sklearn.model_selection import train_test_split

import aspect_based_sentiment_analysis as absa
from aspect_based_sentiment_analysis.training import (
    ClassifierTrainBatch,
    ClassifierDataset,
    EarlyStopping,
    History,
    Logger,
    LossHistory,
    ModelCheckpoint
)


@dataclass
class CategoricalAccuracyHistory(History):
    name: str = 'Accuracy'
    metric: Callable = tf.keras.metrics.CategoricalAccuracy

    @property
    def best_result(self) -> float:
        return max(self.test.values())

    def on_train_batch_end(self, i: int,
                           batch: ClassifierTrainBatch,
                           *train_step_outputs):
        loss_value, logits, *details = train_step_outputs
        acc = self.train_metric(batch.target_labels, logits)
        self.train_details[self.epoch].append(acc.numpy())

    def on_test_batch_end(self, i: int,
                          batch: ClassifierTrainBatch,
                          *test_step_outputs):
        loss_value, logits, *details = test_step_outputs
        acc = self.test_metric(batch.target_labels, logits)
        self.test_details[self.epoch].append(acc.numpy())


def experiment(
        ID: int,
        domain: str,
        base_model_name: str,
        epochs: int,
        batch_size: int = 32,
        learning_rate: float = 3e-5,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        seed: int = 1
) -> float:
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Set up the experiment directory and paths.
    experiment_dir = os.path.join(ROOT_DIR, 'results',
                                  f'classifier-{domain}-{ID:03}')
    os.makedirs(experiment_dir, exist_ok=False)
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    log_path = os.path.join(experiment_dir, 'experiment.log')
    callbacks_path = os.path.join(experiment_dir, 'callbacks.bin')
    # We should remove handlers from the previous experiment, because
    # the logger works on global variables.
    logging.getLogger('absa').handlers = []

    # Load examples from the known labeled datasets like the SemEval. The
    # *test* set is to monitor the training (precisely it's the dev set) and
    # equals 10%.
    examples = absa.load_examples(domain=domain)
    train_examples, test_examples = train_test_split(
        examples, test_size=0.1, random_state=1
    )

    # To build our model, we can define a config, which contains all required
    # information needed to build the `BertABSClassifier` model (including
    # the BERT language model). In this example, we use default parameters
    # (which are set up for our best performance), but of course, you can pass
    # your own parameters (maybe you would be interested to change the number
    # of polarities to classify, or properties of the BERT itself). Moreover, we
    # benefit from the strategy scope to distribute the training. In this
    # case it's only single GPU but the multi GPU training via MirroredStrategy
    # can be used as well.
    strategy = tf.distribute.OneDeviceStrategy('GPU')
    with strategy.scope():
        model = absa.BertABSClassifier.from_pretrained(
            base_model_name,
            output_attentions=True,
            output_hidden_states=True
        )
        tokenizer = transformers.BertTokenizer.from_pretrained(base_model_name)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                             beta_1=beta_1,
                                             beta_2=beta_2,
                                             epsilon=1e-8)

    # To train the model, we have to define a dataset. The dataset can be
    # understood as a non-differential part of the training pipeline. The
    # dataset knows how to transform human-understandable examples into model
    # understandable batches. You are not obligated to use datasets, you can
    # create your own iterable, which transforms classifier example to the
    # classifier train batches.
    dataset = ClassifierDataset(train_examples, batch_size, tokenizer)
    test_dataset = ClassifierDataset(test_examples, batch_size, tokenizer)

    # To easily adjust optimization process to our needs, we define custom
    # training loops called routines (in contrast to use built-in methods as
    # the `fit`). Now, we use the `train_classifier` routine. Each routine
    # has its own optimization step wherein we can control which and how
    # parameters are updated (according to the custom training paradigm
    # presented in the TensorFlow 2.0). We iterate over a dataset, perform
    # train/test optimization steps, and collect results using callbacks
    # (which have a similar interface as the tf.keras.Callback). Please take
    # a look at the `train_classifier` function for more details.
    logger = Logger(file_path=log_path)
    loss_history = LossHistory()
    acc_history = CategoricalAccuracyHistory()
    early_stopping = EarlyStopping(loss_history, patience=3, min_delta=0.001)
    checkpoints = ModelCheckpoint(model, loss_history, checkpoints_dir)
    callbacks = [logger, loss_history, acc_history, checkpoints, early_stopping]
    absa.training.train_classifier(model, optimizer, dataset, epochs,
                                   test_dataset, callbacks, strategy)

    # In the end, we would like to save the model. Our implementation
    # gentle extend the *transformers* lib capabilities, in consequences,
    # `BertABSClassifier` inherits from the `TFBertPreTrainedModel`, and
    # we can do a serialization easily.
    best_model = absa.BertABSClassifier.from_pretrained(checkpoints.best_model_dir)
    best_model.save_pretrained(experiment_dir)
    tokenizer.save_pretrained(experiment_dir)
    absa.utils.save([logger, loss_history, acc_history], callbacks_path)

    # Return the experiment metric value to do the hyper-parameters tuning.
    return acc_history.best_result


def objective(trial, domain: str):
    params = {
        'ID': trial.trial_id,
        'domain': domain,
        'base_model_name': PRETRAINED_MODEL_NAMES[domain],
        'epochs': 20,
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 24, 32]),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-6, 1e-4),
        'beta_1': trial.suggest_uniform('beta_1', 0.5, 1)
    }
    return experiment(**params)


if __name__ == '__main__':
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(ROOT_DIR)
    PRETRAINED_MODEL_NAMES = {
        'restaurant': 'absa/bert-rest-0.1',
        'laptop': 'absa/bert-lapt-0.1'
    }
    for domain in ['restaurant', 'laptop']:
        study = optuna.create_study(
            study_name=f'classifier-{domain}',
            direction='maximize',
            storage='sqlite:///classifier.db',
            load_if_exists=True
        )
        domain_objective = partial(objective, domain=domain)
        study.optimize(domain_objective, n_trials=100)

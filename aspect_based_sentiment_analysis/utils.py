import os
import pickle
import logging
from typing import Any
from typing import Iterable
from typing import List
from google.cloud import storage

logger = logging.getLogger('absa.utils')


def load(file_path: str):
    """ Load arbitrary python objects from the pickled file. """
    with open(file_path, mode='rb') as file:
        return pickle.load(file)


def save(data: Any, file_path: str):
    """ Save arbitrary python objects in the pickled file. """
    with open(file_path, mode='wb') as file:
        pickle.dump(data, file)


def batches(examples: Iterable[Any], batch_size: int,
            reminder: bool = True) -> Iterable[List[Any]]:
    """ Yield an example batch from the example iterable. """
    batch = []
    for example in examples:
        batch.append(example)
        if len(batch) < batch_size:
            continue
        yield batch
        batch = []
    # Return the last incomplete batch if it is necessary.
    if batch and reminder:
        yield batch


def download_from_bucket(bucket_name: str, remote_path: str, local_path: str):
    """ Download the file from the public bucket. """
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)
    blob = storage.Blob(remote_path, bucket)
    blob.download_to_filename(local_path, client=client)


def maybe_download_from_bucket(bucket_name: str, remote_path: str, local_path: str):
    """ Download the file from the bucket if it does not exist. """
    if os.path.isfile(local_path):
        return
    directory = os.path.dirname(local_path)
    os.makedirs(directory, exist_ok=True)
    logger.info('Downloading file from the bucket...')
    download_from_bucket(bucket_name, remote_path, local_path)


def file_from_bucket(name: str):
    """ Load the file stored in the Google Cloud Bucket. """
    bucket = 'aspect-based-sentiment-analysis'
    remote_path = name
    local_path = f'{os.path.dirname(__file__)}/downloads/{name}'
    maybe_download_from_bucket(bucket, remote_path, local_path)
    return local_path


def cache_fixture(fixture):
    """ The function helps to cache test fixtures (only for test cases). """

    def wrapper(request, *args):
        name = request.fixturename
        val = request.config.cache.get(name, None)
        if not val:
            # Make sure that you pass the `request` argument.
            val = fixture(request, *args)
            request.config.cache.set(name, val)
        return val

    return wrapper

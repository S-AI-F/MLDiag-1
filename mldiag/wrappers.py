from typing import Type

import numpy as np
import requests
import tensorflow as tf
import tensorflow_datasets as tfds

from mldiag.services import Service


def txt_tfds_to_numpy(
        dataset: tf.data.Dataset
):
    for datapoint in list(tfds.as_numpy(dataset)):
        data, label = datapoint
        print(data.shape)
        v = np.vectorize(lambda x: x.decode("utf-8"))
        data = v(data)
        yield data, np.asarray(label)


def image_tfds_to_numpy(
        dataset: tf.data.Dataset
):
    for datapoint in list(tfds.as_numpy(dataset)):
        data, label = datapoint
        yield data, np.asarray(label)


def npy_to_numpy(
        dataset_path: str
):
    for datapoint in np.load(dataset_path, allow_pickle=True):
        data, label = datapoint
        v = np.vectorize(lambda x: x.decode("utf-8"))
        data = v(data)
        yield data, np.asarray(label)


def tf_model_to_service(*args, **kwargs) -> Type[Service]:
    service = Service()
    service.predict = (tf.keras.models.load_model(*args, **kwargs)).predict
    return service


def url_to_service(url: str,
                   json_field: str) -> Type[Service]:
    service = Service()

    def fn(x):
        # generator to list
        x = list(x)
        # the list is one tuple of (numpy_array of data, numpy_array of labels)
        x = x[0]  # the first tuple
        x = list(x[0])  # numpy array data to list
        return requests.post(url, json={"text": x}).json()[json_field]

    service.predict = fn
    return service

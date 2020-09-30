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
        v = np.vectorize(lambda x: x.decode("utf-8"))
        data = v(data)
        yield data, np.asarray(label)


def npy_to_numpy(
        dataset_path: str
):
    print(dataset_path)
    for datapoint in np.load(dataset_path, allow_pickle=True):
        data, label = datapoint
        v = np.vectorize(lambda x: x.decode("utf-8"))
        data = v(data)
        yield data, np.asarray(label)


def tf_model_to_service(*args, **kwargs):
    service = Service()
    service.predict = (tf.keras.models.load_model(*args, **kwargs)).predict
    return service


def url_to_service(url):
    service = Service()

    def fn(x):
        x = list(x)
        x = list(x[0][0])
        print(x)
        return requests.post(url, json={"text": x}).json()['results']

    service.predict = fn
    return service

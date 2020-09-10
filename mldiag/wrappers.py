import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def txt_tfds_to_numpy(
        dataset: tf.data.Dataset
):
    for datapoint in list(tfds.as_numpy(dataset)):
        data, label = datapoint
        data = np.asarray(data)
        f = lambda x: x.decode("utf-8")
        v = np.vectorize(f)
        data = v(data)
        yield data, np.asarray(label)

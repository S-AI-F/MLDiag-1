#!/usr/bin/env python

import os
from typing import Dict, Type, Generator

import fire
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import yaml
from tensorflow import keras

from mldiag.services import Service
from mldiag.session import DiagSession
from mldiag.wrappers import image_tfds_to_numpy, tf_model_to_service

module_selection = ("mobilenet_v2_100_224", 224)
handle_base, pixels = module_selection
MODULE_HANDLE = "https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(handle_base)
IMAGE_SIZE = (pixels, pixels)
print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))


def wrap_tfds_as_data(
        custom_config: Dict
) -> Generator:
    (train_data, validation_data, test_data) = tfds.load('tf_flowers',
                                                         split=['train[:70%]',
                                                                'train[70%:80%]',
                                                                'train[80%:]'],
                                                         as_supervised=True)

    def normalize(image, label):
        resized_image = tf.image.resize(image, [pixels, pixels])
        final_image = keras.applications.mobilenet.preprocess_input(resized_image)
        return final_image, label

    test_data = test_data.map(normalize).batch(custom_config["batch_size"])
    return image_tfds_to_numpy(test_data.take(2))


def wrap_tf_model_as_service(model_path: str) -> Type[Service]:
    return tf_model_to_service(
        filepath=model_path,
        custom_objects={'KerasLayer': hub.KerasLayer}
    )


class DiagTextClassification(object):

    def __init__(self):
        # Load a config file
        config_path = os.path.join(os.path.dirname(__file__),
                                   "config_image_classification.yaml")
        with open(config_path) as file:
            self.custom_config = yaml.load(file, Loader=yaml.FullLoader)
        self._default_model_path = os.path.join(os.path.dirname(__file__),
                                                "model.h5")

    def run(self,
            model_path=None,
            report_path=None):
        if model_path is None:
            model_path = self._default_model_path
        # run diag session
        DiagSession(
            config=self.custom_config,
            eval_set=wrap_tfds_as_data(self.custom_config),
            service=wrap_tf_model_as_service(model_path),
            metric=tf.keras.metrics.BinaryAccuracy(),
            report_path=report_path
        ).run()


def main():
    fire.Fire(DiagTextClassification)


if __name__ == '__main__':
    main()

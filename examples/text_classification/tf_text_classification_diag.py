#!/usr/bin/env python

import os
from typing import Dict, Type

import fire
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import yaml
from nptyping import NDArray

from mldiag.services import Service
from mldiag.session import DiagSession
from mldiag.wrappers import txt_tfds_to_numpy, tf_model_to_service


def wrap_tfds_as_data(
        custom_config: Dict
) -> NDArray:
    train_data, validation_data, test_data = tfds.load(
        name="imdb_reviews",
        split=('train[:99%]', 'train[1%:]', 'test'),
        as_supervised=True,
        batch_size=custom_config["batch_size"]
    )
    return txt_tfds_to_numpy(test_data.take(1))


def wrap_tf_model_as_service(model_path: str) -> Type[Service]:
    return tf_model_to_service(
        filepath=model_path,
        custom_objects={'KerasLayer': hub.KerasLayer}
    )


class DiagTextClassification(object):

    def __init__(self):
        # Load a config file
        config_path = os.path.join(os.path.dirname(__file__),
                                   "config_text_classification.yaml")
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

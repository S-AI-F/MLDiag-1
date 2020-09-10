import fire
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

from mldiag.session import DiagSession
from mldiag.wrappers import txt_tfds_to_numpy

# TODO YAML config file
custom_config = {
    "task": "text:classification",  # task type
    "data_type": "text",  # type of data
    "diag_services": "all",  # list of required services
    "model_framework": "tensorflow",  # TODO add version if required, exp: tensorflow:2.x
    "metrics": ["accuracy"],
    "batch_size": 512
}


class DiagTextClassification(object):

    def __init__(self):
        self.train_data, self.validation_data, self.test_data = tfds.load(
            name="imdb_reviews",
            split=('train[:99%]', 'train[1%:]', 'test'),
            as_supervised=True,
            batch_size=custom_config["batch_size"]
        )

        self.eval_set = txt_tfds_to_numpy(self.test_data.take(1))

        pipeline = tf.keras.models.load_model(
            r"C:\Users\SHABOUA\ws\tmp\mldiag\model.h5",
            custom_objects={'KerasLayer': hub.KerasLayer})

        self.predictor = pipeline.predict

        self.metrics = [tf.keras.metrics.BinaryAccuracy()]

    def run(self):
        diag_session = DiagSession(
            config=custom_config,
            eval_set=self.eval_set,
            predictor=self.predictor,
            metrics=self.metrics
        )

        diag_session.run()


if __name__ == '__main__':
    fire.Fire(DiagTextClassification)

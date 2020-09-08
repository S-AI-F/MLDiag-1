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

        self.predictor = pipeline.evaluate

    def run(self):
        diag_session = DiagSession(
            config=custom_config,
            eval_set=self.eval_set,
            predictor=self.predictor,
        )

        diag_session.run()


if __name__ == '__main__':
    # fire.Fire(TextClassification)
    fire.Fire(DiagTextClassification)

# class TextClassification(object):
#     def __init__(self,):
#         self.train_data, self.validation_data, self.test_data = tfds.load(
#             name="imdb_reviews",
#             split=('train[:60%]', 'train[60%:]', 'test'),
#             as_supervised=True)
#
#         embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
#         hub_layer = hub.KerasLayer(
#             embedding,
#             input_shape=[],
#             dtype=tf.string,
#             trainable=True)
#
#         model = tf.keras.Sequential()
#         model.add(hub_layer)
#         model.add(tf.keras.layers.Dense(16, activation='relu'))
#         model.add(tf.keras.layers.Dense(1))
#
#         print (model.summary())
#
#         model.compile(
#             optimizer='adam',
#             loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#             metrics=['accuracy'])
#         self.model = model
#
#     def train(self,
#               save_model_path):
#         history = self.model.fit(
#             self.train_data.shuffle(10000).batch(512),
#             epochs=10,
#             validation_data=self.validation_data.batch(512),
#             verbose=1)
#         self.model.save(os.path.join(save_model_path, "model.h5"))
#
#     def test(self,
#              model_path):
#         model = tf.keras.models.load_model(
#             model_path,
#             custom_objects={'KerasLayer':hub.KerasLayer})
#         print (model.summary())
#
#         results = model.evaluate(
#             self.test_data.batch(512),
#             verbose=2)
#
#         for name, value in zip(model.metrics_names, results):
#             print("%s: %.3f" % (name, value))
#
#
#     def adversial(self,
#                   model_path):
#         model = tf.keras.models.load_model(
#             model_path,
#             custom_objects={'KerasLayer':hub.KerasLayer})
#
#         def augment(dataset, aug):
#             # iterate over all sentences
#             for datapoint in tfds.as_numpy(dataset):
#                 sentence, label = datapoint
#                 new_sentence = aug.augment(sentence.decode('utf-8'))
#                 yield tf.convert_to_tensor(new_sentence), label
#
#         from functools import partial
#         import nlpaug.augmenter.char as nac
#         aug = nac.OcrAug()
#         BATCH = 512
#         test_batches = tf.data.Dataset.from_generator(
#             partial(augment, dataset=self.test_data, aug=aug),
#             output_types=(tf.string, tf.int64),
#             output_shapes=(tf.TensorShape([]), tf.TensorShape([]))
#         ).batch(BATCH)
#         results = model.evaluate(
#             test_batches,
#             verbose=2)
#
#         for name, value in zip(model.metrics_names, results):
#             print("%s: %.3f" % (name, value))

import os

import fire
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class TextClassification(object):
    def __init__(self, ):
        self.train_data, self.validation_data, self.test_data = tfds.load(
            name="imdb_reviews",
            split=('train[:60%]', 'train[60%:]', 'test'),
            as_supervised=True)

        embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
        hub_layer = hub.KerasLayer(
            embedding,
            input_shape=[],
            dtype=tf.string,
            trainable=True)

        model = tf.keras.Sequential()
        model.add(hub_layer)
        model.add(tf.keras.layers.Dense(16, activation='relu'))
        model.add(tf.keras.layers.Dense(2))

        print(model.summary())

        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=['accuracy'])
        self.model = model

    def train(self,
              save_model_path):
        if not os.path.isdir(save_model_path):
            raise ValueError("Missing output directory {} for reporting".format(save_model_path))

        self.model.fit(
            self.train_data.shuffle(10000).batch(512),
            epochs=10,
            validation_data=self.validation_data.batch(512),
            verbose=1)
        self.model.save(os.path.join(save_model_path, "model.h5"))

    def test(self,
             model_path):
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'KerasLayer': hub.KerasLayer})
        print(model.summary())
        pred = model.predict(
            self.test_data.batch(512),
            verbose=2)
        print(pred)

        results = model.evaluate(
            self.test_data.batch(512),
            verbose=2)

        for name, value in zip(model.metrics_names, results):
            print("%s: %.3f" % (name, value))

    def save_test_set(self,
                      out_path,
                      batch_size=512,
                      num_batches=1):
        if not os.path.isdir(out_path):
            raise ValueError("out_path {} not found".format(out_path))

        test_set = np.array(list(tfds.as_numpy(self.test_data.batch(batch_size).take(num_batches))))
        print("test set shape = {}".format(test_set.shape))
        test_set_path = os.path.join(out_path, 'test')
        np.save(test_set_path, test_set)
        print("saved in {}".format(test_set_path + ".npy"))


if __name__ == '__main__':
    fire.Fire(TextClassification)

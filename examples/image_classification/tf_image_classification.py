import os

import fire
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tensorflow import keras

module_selection = ("mobilenet_v2_100_224", 224)
handle_base, pixels = module_selection
MODULE_HANDLE = "https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(handle_base)
IMAGE_SIZE = (pixels, pixels)
print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))

BATCH_SIZE = 32


class TextClassification(object):
    def __init__(self, ):

        (self.train_data, self.validation_data, self.test_data), info = tfds.load('tf_flowers',
                                                                                  split=['train[:70%]',
                                                                                         'train[70%:80%]',
                                                                                         'train[80%:]'],
                                                                                  as_supervised=True,
                                                                                  with_info=True)

        def normalize(image, label):
            resized_image = tf.image.resize(image, [pixels, pixels])
            final_image = keras.applications.mobilenet.preprocess_input(resized_image)
            return final_image, label

        self.test_data = self.test_data.map(normalize).batch(BATCH_SIZE).prefetch(1)
        self.validation_data = self.validation_data.map(normalize).batch(BATCH_SIZE).prefetch(1)
        self.train_data = self.train_data.map(normalize).batch(BATCH_SIZE).prefetch(1)

        do_fine_tuning = False

        print("Building model with", MODULE_HANDLE)
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
            hub.KerasLayer(MODULE_HANDLE, trainable=do_fine_tuning),
            tf.keras.layers.Dense(512),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(info.features['label'].num_classes)
        ])
        model.build((None,) + IMAGE_SIZE + (3,))
        model.summary()

        print(model.summary())

        model.compile(
            optimizer=tf.keras.optimizers.SGD(lr=0.01),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy'])

        self.model = model

    def train(self,
              save_model_path):
        if not os.path.isdir(save_model_path):
            raise ValueError("Missing output directory {} for reporting".format(save_model_path))

        self.model.fit(
            self.train_data,
            epochs=3,
            validation_data=self.validation_data,
            verbose=1)
        self.model.save(os.path.join(save_model_path, "model.h5"))

    def test(self,
             model_path):
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'KerasLayer': hub.KerasLayer})
        print(model.summary())

        pred = model.predict(
            self.test_data,
            verbose=2)
        print(pred)

        results = model.evaluate(
            self.test_data,
            verbose=2)

        for name, value in zip(model.metrics_names, results):
            print("%s: %.3f" % (name, value))

    def save_test_set(self,
                      out_path,
                      batch_size=BATCH_SIZE,
                      num_batches=16):
        if not os.path.isdir(out_path):
            raise ValueError("out_path {} not found".format(out_path))

        lst = list(tfds.as_numpy(self.test_data.take(num_batches)))

        # TODO: actually we concatenate all batches in one, since mldiag take only one batch..
        # Improve MLDIAG to take multiple batches
        images = np.concatenate([x[0] for x in lst])
        images = np.array([x.tobytes() for x in images])
        labels = np.concatenate([x[1] for x in lst])
        test_set = np.array([(images, labels)])

        # test_set = np.array(list(tfds.as_numpy(self.test_data.map(create_images).batch(batch_size).take(
        # num_batches))))

        print("test set shape = {}".format(test_set.shape))
        test_set_path = os.path.join(out_path, 'test')
        np.save(test_set_path, test_set)
        print("saved in {}".format(test_set_path + ".npy"))


if __name__ == '__main__':
    fire.Fire(TextClassification)

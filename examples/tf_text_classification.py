import os

import fire
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

    def adversial(self,
                  model_path):
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'KerasLayer': hub.KerasLayer})

        def augment(dataset, aug):
            # iterate over all sentences
            for datapoint in tfds.as_numpy(dataset):
                sentence, label = datapoint
                new_sentence = aug.augment(sentence.decode('utf-8'))
                yield tf.convert_to_tensor(new_sentence), label

        from functools import partial
        import nlpaug.augmenter.char as nac
        aug = nac.OcrAug()
        BATCH = 512
        test_batches = tf.data.Dataset.from_generator(
            partial(augment, dataset=self.test_data, aug=aug),
            output_types=(tf.string, tf.int64),
            output_shapes=(tf.TensorShape([]), tf.TensorShape([]))
        ).batch(BATCH)
        results = model.evaluate(
            test_batches,
            verbose=2)

        for name, value in zip(model.metrics_names, results):
            print("%s: %.3f" % (name, value))


if __name__ == '__main__':
    fire.Fire(TextClassification)

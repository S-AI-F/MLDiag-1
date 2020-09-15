from typing import Dict

import numpy as np
import tensorflow as tf


class Metric(object):
    def __init__(self,
                 tf_metric: tf.keras.metrics.Metric):
        """

        :param tf_metric:
        """
        if not isinstance(tf_metric, tf.keras.metrics.Metric):
            raise TypeError("Only tensorflow Metric object is supported")
        self.tf_metric = tf_metric

    def eval(self,
             y_true: np.ndarray,
             y_pred: np.ndarray) -> Dict:
        """

        :param y_true:
        :param y_pred:
        :return:
        """

        length = len(y_true)
        if length != len(y_pred):
            raise ValueError("Vectors have different lengths: true {} vs pred {}".format(length, len(y_pred)))
        if y_true.shape[1] != 1:
            raise ValueError("Shape of true labels vector should be ({},1)".format(length))
        if y_pred.shape[1] != 1:
            raise ValueError("Shape of pred labels vector should be ({},1)".format(length))

        self.tf_metric.update_state(y_true, y_pred)

        return {"Metric": self.tf_metric.__class__.__name__, "Result": self.tf_metric.result().numpy()}

import numpy as np
from nlpaug.base_augmenter import Augmenter


def augment(
        dataset,
        augmenter: Augmenter,
):
    for datapoint in dataset:
        data, label = datapoint
        data = np.asarray(data)
        data = augmenter.augments(data)
        yield np.asarray(data), np.asarray(label)

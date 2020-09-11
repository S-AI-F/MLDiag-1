from functools import partial

import nlpaug.augmenter.char as nac
import numpy as np
from nlpaug.base_augmenter import Augmenter


def augment(
        dataset,
        augmenter: Augmenter,
):
    for datapoint in dataset:
        data, label = datapoint
        if augmenter is not None:
            data = augmenter.augments(list(data))
        yield np.asarray(data), np.asarray(label)


none_augmenter = partial(augment, augmenter=None)
text_ocr_augmnter = partial(augment, augmenter=nac.OcrAug())

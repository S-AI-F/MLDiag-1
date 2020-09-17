from functools import partial

import nlpaug.augmenter.char as nac
import numpy as np
from nlpaug.base_augmenter import Augmenter


def augment(
        dataset: list,
        augmenter: Augmenter,
):
    """
    a generic augment process on generator
    :param dataset: dataset generator (batch)
    :param augmenter:
    :return:
    """
    for data_point in dataset:
        data, label = data_point
        if augmenter is not None:
            data = augmenter.augments(list(data))
        yield np.asarray(data), np.asarray(label)


# NULL augmenter
none_augmenter = partial(augment, augmenter=None)

# OCR error augmenter
text_ocr_augmnter = partial(augment, augmenter=nac.OcrAug())

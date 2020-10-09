from functools import partial

import nlpaug.augmenter.char as nac
import numpy as np
from imgaug import augmenters as iaa
from nlpaug.base_augmenter import Augmenter


def augment_text(
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


def augment_image(
        dataset: list,
        augmenter,
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
            data = augmenter(images=data)
        yield np.asarray(data), np.asarray(label)


# NULL augmenter
none_augmenter = partial(augment_text, augmenter=None)

# OCR error augmenter
text_ocr_augmenter = partial(augment_text, augmenter=nac.OcrAug())

# flip_r image augmenter
image_flip_r_augmenter = partial(augment_image, augmenter=iaa.Fliplr())

# rotation image augmenter
image_rot_augmenter = partial(augment_image, augmenter=iaa.Rotate())

from abc import ABC, abstractmethod
from functools import partial
from typing import List

import numpy as np


# TODO: reproduce the Augmenter base class of nlpaug
class Description(ABC):
    def __init__(self,
                 name,
                 verbose):
        self.name = name
        self.verbose = verbose

    @abstractmethod
    def describe(self,
                 data):
        raise NotImplementedError


    def describes(self,
                  data: List):
        out = []
        for d in data:
            out.append(self.describe(d))
        return out


def gen_describe(
        dataset: list,
        description: Description,
):
    """
    a generic augment process on generator
    :param dataset: dataset generator [batch]
    :param augmenter:
    :return:
    """
    for data_point in dataset:
        data, label = data_point
        descriptors = np.empty(len(list(data)))
        if description is not None:
            descriptors = description.describes(list(data))
        else:
            descriptors = np.nan
        yield np.asarray(descriptors)


class TextCharCountDescription(Description):
    def __init__(self, name="text_char_count", verbose=0):
        super().__init__(
            name=name,
            verbose=verbose)

    def describe(self,
                 data):
        if not isinstance(data, str):
            raise TypeError("data should be string and not {}".format(type(data)))
        return len(data)


class TextSentenceCountDescription(Description):
    def __init__(self, name="text_sentence_count", verbose=0):
        super().__init__(
            name=name,
            verbose=verbose)

    def describe(self,
                 data):
        if not isinstance(data, str):
            raise TypeError("data should be string and not {}".format(type(data)))
        return len(data.split("."))


class TextWordCountDescription(Description):
    def __init__(self, name="text_word_count", verbose=0):
        super().__init__(
            name=name,
            verbose=verbose)

    def describe(self,
                 data):
        if not isinstance(data, str):
            raise TypeError("data should be string and not {}".format(type(data)))
        return sum([len(x.split(" ")) for x in data.split(".")])


# text char count description
text_char_count_descriptor = partial(gen_describe, description=TextCharCountDescription())
# text sentence count description
text_sentence_count_descriptor = partial(gen_describe, description=TextSentenceCountDescription())
# text word count description
text_word_count_descriptor = partial(gen_describe, description=TextWordCountDescription())

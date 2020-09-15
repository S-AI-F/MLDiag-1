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

    @abstractmethod
    def describes(self,
                  data: List):
        raise NotImplementedError


def gen_describe(
        dataset: list,
        description: Description,
):
    """
    a generic augment process on generator
    :param dataset: dataset generator
    :param augmenter:
    :return:
    """
    for data_point in dataset:
        data, label = data_point
        descriptors = np.empty(len(list(data)))
        if description is not None:
            descriptors = description.describe(list(data))
        else:
            descriptors = np.nan
        yield np.asarray(descriptors)


class TextLengthDescription(Description):
    def __init__(self, name="text_len", verbose=0):
        super().__init__(
            name=name,
            verbose=verbose)

    def describe(self,
                 data):
        if not isinstance(data, str):
            raise TypeError("data should be string and not {}".format(type(data)))
        return len(data)

    def describes(self,
                  data: List) -> List:
        out = []
        for d in data:
            out.append(self.describe(d))
        return out


# Length text description
text_len_descriptor = partial(gen_describe, description=TextLengthDescription())

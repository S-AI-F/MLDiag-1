from typing import List, Dict

from mldiag import augmenters, descriptors


class Method:
    # None method to return the original dataset used to compare results obtained by other augmenters
    NONE = {
        "type": "augment",
        "name": "none",
        "fn": augmenters.none_augmenter
    }

    @staticmethod
    def get_one(method: str) -> List[Dict]:
        '''
        get one method dict
        :param method: the method name
        :return: list(dict)
        '''

        # all possible methods
        if method == "all":
            return [m for m in Method.get_all() if m['fn']]

        # the none/null method
        if method == "none":
            return [Method.NONE]

        # find a specific method
        for m in Method.get_all():
            if m['name'] == method:
                return [m]

    @staticmethod
    def get_all() -> List[Dict]:
        """
        get all methods
        :return:
        """
        return TextMethod.get_all() + ImageMethod.get_all()

    @staticmethod
    def get_all_names() -> List[str]:
        """
        get all method names
        :return:
        """
        return TextMethod.get_all_names() + ["none", "all"] + ImageMethod.get_all_names()

    @staticmethod
    def check(method: str) -> str:
        """
        check method name in the list of possible methods
        :param method:
        :return:
        """
        if not (method in Method.get_all_names()):
            raise ValueError(
                'Method must be one of {} while {} is passed'.format(Method.get_all_names(), method))
        return method


class TextMethod:
    """
    Text methods only
    """

    # augment a text with ocr errors
    METHOD_CHAR_OCR = {
        "type": "augment",
        "name": "char_ocr",
        "fn": augmenters.text_ocr_augmenter
    }

    METHOD_TXT_CHAR_COUNT = {
        "type": "describe",
        "name": "text_char_count",
        "fn": descriptors.text_char_count_descriptor
    }

    METHOD_TXT_SENTENCE_COUNT = {
        "type": "describe",
        "name": "text_sentence_count",
        "fn": descriptors.text_sentence_count_descriptor
    }

    METHOD_TXT_WORD_COUNT = {
        "type": "describe",
        "name": "text_word_count",
        "fn": descriptors.text_word_count_descriptor
    }

    @staticmethod
    def get_all() -> List[Dict]:
        """
        get all text methods
        :return:
        """
        return [TextMethod.__dict__[m] for m in TextMethod.__dict__.keys() if m.startswith("METHOD_")]

    @staticmethod
    def get_all_names() -> List[str]:
        """
        get all text method names
        :return:
        """
        return [m["name"] for m in TextMethod.get_all()]


class ImageMethod:
    """
    Text methods only
    """

    # augment an imge with filp
    METHOD_IMAGE_FLIP_R = {
        "type": "augment",
        "name": "image_flip_r",
        "fn": augmenters.image_flip_r_augmenter
    }

    # augment an imge with filp
    METHOD_IMAGE_ROTATE = {
        "type": "augment",
        "name": "image_rotate",
        "fn": augmenters.image_rot_augmenter
    }

    @staticmethod
    def get_all() -> List[Dict]:
        """
        get all text methods
        :return:
        """
        return [ImageMethod.__dict__[m] for m in ImageMethod.__dict__.keys() if m.startswith("METHOD_")]

    @staticmethod
    def get_all_names() -> List[str]:
        """
        get all text method names
        :return:
        """
        return [m["name"] for m in ImageMethod.get_all()]

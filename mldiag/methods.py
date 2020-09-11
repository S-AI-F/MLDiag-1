import itertools

from mldiag import augmentors


class Method:
    NONE = {
        "name": "none",
        "fn": augmentors.none_augmenter
    }

    @staticmethod
    def get_one(method):
        if method == "all":
            return [m for m in Method.get_all() if m['fn']]
        if method == "none":
            return [Method.NONE]
        for m in Method.get_all():
            if m['name'] == method:
                return [m]

    @staticmethod
    def get_all():
        return list(itertools.chain(*[TextMethod.get_all()]))

    @staticmethod
    def get_all_names():
        return list(itertools.chain(*[TextMethod.get_all_names()])) + ["none", "all"]

    @staticmethod
    def check(method: str):
        if not (method in Method.get_all_names()):
            raise ValueError(
                'Method must be one of {} while {} is passed'.format(Method.get_all_names(), method))


class TextMethod:
    METHOD_CHAR_OCR = {
        "name": "char_ocr",
        "fn": augmentors.text_ocr_augmnter
    }

    @staticmethod
    def get_all():
        return [TextMethod.__dict__[m] for m in TextMethod.__dict__.keys() if m.startswith("METHOD_")]

    @staticmethod
    def get_all_names():
        return [m["name"] for m in TextMethod.get_all()]

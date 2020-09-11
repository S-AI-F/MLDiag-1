import itertools

from mldiag import methods


class GenericRunner(object):

    def __init__(self, method):
        if not (method in methods.Method.get_all_names()):
            raise ValueError(
                'Method must be one of {} while {} is passed'.format(methods.Method.get_all_names(), method))
        self.method = method

    def get_all(self):
        return methods.Method.get_one(self.method)


class Runners():

    @staticmethod
    def get_all():
        return TextRunners.get_all()


class TextRunners():

    @staticmethod
    def get_all():
        return TextClassificationRunners.get_all()


class TextClassificationRunners(object):

    @staticmethod
    def get_all():
        return list(itertools.chain(*[GenericRunner(method=m).get_all() for m in methods.TextMethod.get_all_names()]))

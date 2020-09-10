import itertools
import logging

import numpy as np

from mldiag import runners

log = logging.getLogger(__name__)


class DiagSession(object):
    def __init__(self,
                 config: dict,
                 eval_set,
                 predictor,
                 metrics):
        self.config = config
        # TODO: solve the problem of generator clones to use a generator multipletime
        # from itertools import tee
        # self.eval_set, self.true = tee(eval_set)
        self.eval_set = list(eval_set)
        # self.true = list(eval_set)

        self.predictor = predictor
        self.metrics = metrics

    def _prepare_runners(self):
        runs = []

        if self.config["diag_services"] == "all":
            macro_task, micro_task = self.config['task'].split(sep=":")

            if macro_task == "text":
                if micro_task == "classification":
                    import nlpaug.augmenter.char as nac
                    runs.append(
                        runners.augment(
                            dataset=list(self.eval_set),
                            augmenter=nac.OcrAug(),
                        )
                    )
        return runs

    def _make_run(
            self,
            runner):

        pred = self.predictor(
            runner,
            verbose=2)
        pred = np.argmax(pred, axis=-1)
        y_true = np.array(list(itertools.chain([x[1] for x in self.eval_set]))[0])

        for metric in self.metrics:
            metric.update_state(y_true.reshape(-1, 1), pred.reshape(-1, 1))
            result = metric.result().numpy()
            print(result)

    def run(self):
        runs = self._prepare_runners()
        for runner in runs:
            self._make_run(runner)

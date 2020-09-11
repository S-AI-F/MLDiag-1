import itertools
import logging

import numpy as np

from mldiag import report
from mldiag import runners

log = logging.getLogger(__name__)


class DiagSession(object):
    def __init__(self,
                 config: dict,
                 eval_set,
                 predictor,
                 metrics):
        self.config = config
        self.eval_set = list(eval_set)
        self.predictor = predictor
        self.metrics = metrics

    def _prepare_runners(self):
        # None runner: run the model on the current dataset without changes
        runs = runners.GenericRunner(method="none", task="none:none").get_all()

        if self.config["diag_services"] == "all":
            # add all runners
            runs += runners.Runners(task=self.config['task']).get_all()
        else:
            # add specific runners
            for run_f in self.config["diag_services"]:
                runs += runners.GenericRunner(method=run_f, task=self.config['task']).get_all()

        return runs

    def _make_run(
            self,
            runner):

        pred = self.predictor(
            runner['fn'](dataset=self.eval_set)
        )
        pred = np.argmax(pred, axis=-1)
        y_true = np.array(list(itertools.chain(*[x[1] for x in self.eval_set])))
        results = []
        for metric in self.metrics:
            metric.update_state(y_true.reshape(-1, 1), pred.reshape(-1, 1))
            results.append(
                {"Method": runner['name'], "Metric": metric.__class__.__name__, "Result": metric.result().numpy()})
        return results

    def run(self):

        runs = self._prepare_runners()
        results = []
        for runner in runs:
            results += self._make_run(runner)

        rep = report.DiagReport(dict_results=results)
        rep.out()

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
        runs = runners.GenericRunner("none").get_all()

        macro_task, micro_task = self.config['task'].split(sep=":")

        if self.config["diag_services"] == "all":

            if macro_task == "text":
                if micro_task == "classification":
                    runs += runners.TextClassificationRunners.get_all()
        else:
            for run_f in self.config["diag_services"]:
                # TODO check if the diag method is possible regarding the task
                runs += runners.GenericRunner(run_f).get_all()

        return runs

    def _make_run(
            self,
            runner):

        pred = self.predictor(
            runner['fn'](dataset=self.eval_set),
            verbose=2)
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

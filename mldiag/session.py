import itertools
import logging
from typing import Generator, List, Dict

import numpy as np

from mldiag import report, runners

log = logging.getLogger(__name__)


class DiagSession(object):
    """
    Diagnostic session
    """

    def __init__(self,
                 config: dict,
                 eval_set: Generator,
                 predictor,
                 metrics: List):
        self.config = config
        self.eval_set = list(eval_set)
        self.predictor = predictor
        self.metrics = metrics

    def _prepare_runners(self) -> List:
        """
        prepare the lazy runners
        :return:
        """
        # None runner: run the model on the current dataset without changes
        runs = runners.GenericRunner(method="none", task="none:none").get_one()

        if self.config["diag_services"] == "all":
            # add all runners
            runs += runners.GenericRunner.get_all(task=self.config['task'])
        else:
            # add specific runners
            for run_f in self.config["diag_services"]:
                runs += runners.GenericRunner(method=run_f, task=self.config['task']).get_one()

        return runs

    def _make_run(
            self,
            runner: Dict) -> List[Dict]:
        '''
        Make one run
        :param runner: the runner on the eval set
        :return: list of runners results
        '''

        pred = self.predictor(
            runner['fn'](dataset=self.eval_set)
        )
        pred = np.argmax(pred, axis=-1)

        # TODO: check when there is no batch for chain list
        y_true = np.array(list(itertools.chain(*[x[1] for x in self.eval_set])))

        results = []
        for metric in self.metrics:
            # TODO: define custom metric object
            metric.update_state(y_true.reshape(-1, 1), pred.reshape(-1, 1))
            result = {"Method": runner['name'],
                      "Metric": metric.__class__.__name__,
                      "Result": metric.result().numpy()}
            results.append(result)

        return results

    def run(self):
        '''
        Run the dignostic session
        :return:
        '''

        # prepare the runs: lazy functions
        runs = self._prepare_runners()

        results = []
        for runner in runs:
            results += self._make_run(runner)

        # make report
        rep = report.DiagReport(dict_results=results)
        rep.out()

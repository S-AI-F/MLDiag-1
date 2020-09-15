import itertools
import logging
from typing import Generator, List, Dict

import numpy as np

from mldiag import report, runners, metrics

log = logging.getLogger(__name__)


class DiagSession(object):
    """
    Diagnostic session
    """

    def __init__(self,
                 config: dict,
                 eval_set: Generator,
                 predictor,
                 metric):
        self.config = config
        self.eval_set = list(eval_set)
        self.predictor = predictor
        self.metric = metrics.Metric(metric)

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

    def _make_augment_run(self,
                          runner: Dict) -> Dict:
        pred = self.predictor(
            runner['fn'](dataset=self.eval_set)
        )
        pred = np.argmax(pred, axis=-1)

        # TODO: check when there is no batch for chain list
        y_true = np.array(list(itertools.chain(*[x[1] for x in self.eval_set])))

        result = self.metric.eval(
            y_true=y_true.reshape(-1, 1),
            y_pred=pred.reshape(-1, 1)
        )

        return {"Type": "augment",
                "Method": runner['name'],
                "Metric": result['Metric'],
                "Result": result['Result']}

    def _make_describe_run(self,
                           runner: Dict) -> Dict:
        out = runner['fn'](dataset=self.eval_set)
        print(out)
        return {"Type": "describe"}

    def _make_run(
            self,
            runner: Dict) -> Dict:
        '''
        Make one run
        :param runner: the runner on the eval set
        :return: list of runners results
        '''

        if runner['type'] == "augment":
            return self._make_augment_run(runner)
        if runner['type'] == "describe":
            return self._make_describe_run(runner)
        raise ValueError("Unknown runner type {}, should be in {}".format(runner['type'], ["augment", "describe"]))


    def run(self):
        '''
        Run the dignostic session
        :return:
        '''

        # prepare the runs: lazy functions
        runs = self._prepare_runners()

        results = []
        for runner in runs:
            results.append(self._make_run(runner))

        # make report
        rep = report.DiagReport(dict_results=results)
        rep.out()

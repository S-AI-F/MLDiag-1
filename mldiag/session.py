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
        self.original_y_pred = None  # predictions of the model on the original eval set
        self.y_true = np.array(list(itertools.chain(*[x[1] for x in self.eval_set])))

    def _prepare_runners(self) -> List:
        """
        prepare the lazy runners
        :return:
        """
        runs = []
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
        if runner["name"] == "none":
            self.original_y_pred = pred

        # TODO: check when there is no batch for chain list

        result = self.metric.eval(
            y_true=self.y_true.reshape(-1, 1),
            y_pred=pred.reshape(-1, 1)
        )

        return {"Type": "augment",
                "Method": runner['name'],
                "Metric": result['Metric'],
                "Result": result['Result']}

    def _make_describe_run(self,
                           runner: Dict) -> Dict:
        out = runner['fn'](dataset=self.eval_set)

        # where errors occurs
        def errors(y_true, y_pred, desc):
            return (y_true == y_pred, desc)

        res = list(map(errors, self.y_true, self.original_y_pred, list(out)[0]))

        ko = list([r[1] for r in res if r[0] == False])
        ok = list([r[1] for r in res if r[0] == True])

        density_ok, bins = np.histogram(ok, normed=False, density=True)
        unity_density_ok = density_ok / density_ok.sum()
        hist_ok = ["({:.0%},{})".format(x, int(y)) for x, y in zip(unity_density_ok, bins)]

        density_ko, _ = np.histogram(ko, normed=False, density=True, bins=bins)
        unity_density_ko = density_ko / density_ko.sum()
        hist_ko = ["({:.0%},{})".format(x, int(y)) for x, y in zip(unity_density_ko, bins)]

        hist_diff = [(x - y) for x, y in zip(unity_density_ko, unity_density_ok)]
        hist_diff = ["({:.0%},{})".format(x, int(y)) for x, y in zip(hist_diff, bins)]

        return {"Type": "describe",
                "Method": runner['name'],
                "Metric": "histogram",
                "Result": (hist_ko, hist_ok, hist_diff)}

    def run(self):
        '''
        Run the dignostic session
        :return:
        '''

        # None runner: run the model on the current dataset without changes
        runner_0 = runners.GenericRunner(method="none", task="none:none").get_one()
        results_augment = [self._make_augment_run(runner_0[0])]

        results_describe = []
        # prepare the runs: lazy functions
        runs = self._prepare_runners()
        for runner in runs:
            if runner['type'] == "augment":
                results_augment.append(self._make_augment_run(runner))
            elif runner['type'] == "describe":
                results_describe.append(self._make_describe_run(runner))
            else:
                raise ValueError(
                    "Unknown runner type {}, should be in {}".format(runner['type'], ["augment", "describe"]))

        # make report
        rep = report.DiagReport(
            dict_results_augment=results_augment,
            dict_results_describe=results_describe)
        rep.out()
        rep.html(html_file_path=self.config["report_file_path"])

import logging

from mldiag import runner

log = logging.getLogger(__name__)


class DiagSession(object):
    def __init__(self,
                 config: dict,
                 eval_set,
                 predictor, ):
        self.config = config
        self.eval_set = eval_set
        self.predictor = predictor

    def _prepare_runners(self):
        runners = []

        if self.config["diag_services"] == "all":
            macro_task, micro_task = self.config['task'].split(sep=":")

            if macro_task == "text":
                if micro_task == "classification":
                    import nlpaug.augmenter.char as nac
                    runners.append(
                        runner.augment(
                            dataset=self.eval_set,
                            augmenter=nac.OcrAug(),
                        )
                    )
        return runners

    def _make_run(
            self,
            runner):

        results = self.predictor(
            runner,
            verbose=2)
        print(results)

    def run(self):

        runners = self._prepare_runners()

        for runner in runners:
            self._make_run(runner)

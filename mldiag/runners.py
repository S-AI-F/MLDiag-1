from typing import List, Dict

from mldiag import methods, tasks, diagnostics


class GenericRunner(object):
    """
    A generic runner on an eval set
    """

    def __init__(self,
                 task: str,
                 method: str):
        self.method = methods.Method.check(method)
        self.macro_task, self.micro_task = tasks.Task.check(task)
        diagnostics.Diagnostics.check(task, method)

    def get_one(self) -> List[Dict]:
        return methods.Method.get_one(self.method)

    @staticmethod
    def get_all(task: str) -> List[Dict]:
        macro_task, micro_task = tasks.Task.check(task)
        runs = []
        for diag_tuple in diagnostics.Diagnostics.ACCEPTED:
            if diag_tuple[0] == macro_task and diag_tuple[1] == micro_task:
                runs += GenericRunner(method=diag_tuple[2], task=task).get_one()
        return runs

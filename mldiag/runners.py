import itertools

from mldiag import methods, tasks, diagnostics


class GenericRunner(object):

    def __init__(self, task: str, method: str):
        methods.Method.check(method)
        tasks.Task.check(task)
        diagnostics.Diagnostics.check(task, method)
        self.method = method

    def get_all(self):
        return methods.Method.get_one(self.method)


class Runners(object):

    def __init__(self, task: str):
        self.macro_task, self.micro_task = tasks.Task.check(task)
        self.task = task

    def get_all(self):
        if self.macro_task == tasks.MacroTask.MACRO_TASK_TXT:
            if self.micro_task == tasks.MicroTask.MICRO_TASK_CLASSIFICATION:
                return list(itertools.chain(*[
                    GenericRunner(method=m, task=self.task).get_all() for m in methods.TextMethod.get_all_names()
                ]))

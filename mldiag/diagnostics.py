from mldiag import methods, tasks


class Diagnostics(object):
    ACCEPTED = [
        (tasks.Task.NONE, tasks.Task.NONE, methods.Method.NONE["name"]),
        (tasks.MacroTask.MACRO_TASK_TXT, tasks.MicroTask.MICRO_TASK_CLASSIFICATION,
         methods.TextMethod.METHOD_CHAR_OCR["name"])
    ]

    @staticmethod
    def check(task: str, method: str):
        tpl = tuple(tasks.Task.split(task) + [method])
        if tpl not in Diagnostics.ACCEPTED:
            raise ValueError(
                'Task/Method must be one of {} while {} is passed'.format(Diagnostics.ACCEPTED, tpl))

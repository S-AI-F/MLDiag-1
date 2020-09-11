class MacroTask(object):
    MACRO_TASK_TXT = "text"
    MACRO_TASK_IMAGE = "image"

    @staticmethod
    def get_all():
        return [MacroTask.__dict__[m] for m in MacroTask.__dict__.keys() if m.startswith("MACRO_TASK_")]

    @staticmethod
    def check(macro_task):
        if macro_task not in MacroTask.get_all():
            raise ValueError(
                'Macro task must be one of {} while {} is passed'.format(MacroTask.get_all(), macro_task))
        return macro_task


class MicroTask(object):
    MICRO_TASK_CLASSIFICATION = "classification"

    @staticmethod
    def get_all():
        return [MicroTask.__dict__[m] for m in MicroTask.__dict__.keys() if m.startswith("MICRO_TASK_")]

    @staticmethod
    def check(micro_task):
        if micro_task not in MicroTask.get_all():
            raise ValueError(
                'Macro task must be one of {} while {} is passed'.format(MicroTask.get_all(), micro_task))
        return micro_task


class Task(object):
    NONE = "none"

    @staticmethod
    def split(task: str):
        return task.split(":")[:2]

    @staticmethod
    def check(task: str):
        macro_task, micro_task = Task.split(task)
        if macro_task == "none" and micro_task == "none":
            return macro_task, micro_task
        return MacroTask.check(macro_task), MicroTask.check(micro_task)

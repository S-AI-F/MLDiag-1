from mldiag import methods, tasks


class Diagnostics(object):
    """
    List of accepted diagnostics: tuples of macro_task, micro_task, method
    example: (text, classification, char_ocr)
    """
    ACCEPTED = [
        (tasks.Task.NONE,
         tasks.Task.NONE,
         methods.Method.NONE["name"]),
        (tasks.MacroTask.MACRO_TASK_TXT,
         tasks.MicroTask.MICRO_TASK_CLASSIFICATION,
         methods.TextMethod.METHOD_CHAR_OCR["name"]),
        (tasks.MacroTask.MACRO_TASK_TXT,
         tasks.MicroTask.MICRO_TASK_CLASSIFICATION,
         methods.TextMethod.METHOD_TXT_CHAR_COUNT["name"]),
        (tasks.MacroTask.MACRO_TASK_TXT,
         tasks.MicroTask.MICRO_TASK_CLASSIFICATION,
         methods.TextMethod.METHOD_TXT_SENTENCE_COUNT["name"]),
        (tasks.MacroTask.MACRO_TASK_TXT,
         tasks.MicroTask.MICRO_TASK_CLASSIFICATION,
         methods.TextMethod.METHOD_TXT_WORD_COUNT["name"]),
        (tasks.MacroTask.MACRO_TASK_IMAGE,
         tasks.MicroTask.MICRO_TASK_CLASSIFICATION,
         methods.ImageMethod.METHOD_IMAGE_FLIP_R["name"]),
        (tasks.MacroTask.MACRO_TASK_IMAGE,
         tasks.MicroTask.MICRO_TASK_CLASSIFICATION,
         methods.ImageMethod.METHOD_IMAGE_ROTATE["name"])
    ]

    @staticmethod
    def check(task: str, method: str):
        """
        Check coherence between task and method
        :param task:
        :param method:
        :return:
        """
        tpl = tuple(tasks.Task.split(task) + [method])
        if tpl not in Diagnostics.ACCEPTED:
            raise ValueError(
                'Task/Method must be one of {} while {} is passed'.format(Diagnostics.ACCEPTED, tpl))

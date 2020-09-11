from typing import List, Dict

from termcolor import colored


class DiagReport(object):
    """
    Diagnostic report
    """

    def __init__(self,
                 dict_results: List[Dict]):
        """

        :param dict_results:
        """
        self.dict_results = dict_results

    def out(self):
        """
        terminal output
        :return:
        """
        print(colored("Method", "yellow"),
              "\t", colored("Metric", "blue"),
              "\t", colored("Result", "grey"),
              "\t", colored("Impact", "grey"))
        print("=======================================")
        dict_result_0 = self.dict_results[0]
        print(colored(dict_result_0["Method"], "yellow"),
              "\t", colored(dict_result_0["Metric"], "blue"),
              "\t", colored(dict_result_0["Result"], "grey"),
              "\t", colored("-", "grey"))

        for dict_result in self.dict_results[1:]:
            diff_result = dict_result["Result"] - dict_result_0["Result"]
            print(colored(dict_result["Method"], "yellow"),
                  "\t", colored(dict_result["Metric"], "blue"),
                  "\t", colored(dict_result["Result"], "grey"),
                  "\t", colored(diff_result, "red") if diff_result < 0 else colored(diff_result, "green"))

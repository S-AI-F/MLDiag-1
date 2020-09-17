from typing import List, Dict

from termcolor import colored
from yattag import Doc


class DiagReport(object):
    """
    Diagnostic report
    """

    def __init__(self,
                 dict_results_augment: List[Dict],
                 dict_results_describe: List[Dict]):
        """

        :param dict_results:
        """
        self.dict_results_augment = dict_results_augment
        self.dict_results_describe = dict_results_describe

    def out(self):
        """
        terminal output
        :return:
        """
        print("Invariance report ...")
        print("======================")
        print(colored("Method", "yellow"),
              "\t", colored("Metric", "blue"),
              "\t", colored("Result", "grey"),
              "\t", colored("Impact", "grey"))
        print("=======================================")
        dict_result_0 = self.dict_results_augment[0]
        print(colored(dict_result_0["Method"], "yellow"),
              "\t", colored(dict_result_0["Metric"], "blue"),
              "\t", colored(dict_result_0["Result"], "grey"),
              "\t", colored("-", "grey"))
        for dict_result in self.dict_results_augment[1:]:
            diff_result = dict_result["Result"] - dict_result_0["Result"]
            print(colored(dict_result["Method"], "yellow"),
                  "\t", colored(dict_result["Metric"], "blue"),
                  "\t", colored(dict_result["Result"], "grey"),
                  "\t", colored(diff_result, "red") if diff_result < 0 else colored(diff_result, "green"))

        print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("Descriptive report ...")
        print("======================")
        print(colored("Method", "yellow"),
              "\t", colored("Metric", "blue"),
              "\t", colored("Result", "grey"),
              "\t", colored("Impact", "grey"))
        print("=======================================")
        for dict_result in self.dict_results_describe:
            print(colored(dict_result["Method"], "yellow"),
                  "\t", colored(dict_result["Metric"], "blue"),
                  "\t", dict_result["Result"][0],
                  "\t", dict_result["Result"][2], )

    def html(self,
             html_file_path):
        doc, tag, text = Doc().tagtext()

        with tag('html'):
            with tag('body'):
                text("Welcome to our site")
                doc.stag('img', src='file:../resources/ml-diag.jpg', width="200", height="200")

        print(doc.getvalue())
        with open(html_file_path, "w") as text_file:
            text_file.write(doc.getvalue())
        '''
        # Add main header
        document.add_header('ML Diag', level='h1', align='center')
        # load the image
        image = Image.open('../resources/ml-diag.png').convert("RGB")
        # convert image to numpy array
        data = asarray(image)
        '''

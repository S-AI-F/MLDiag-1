import base64
import pathlib
from io import BytesIO
from typing import List, Dict, Union

import PIL
import numpy as np
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
                doc.stag('img', src=self._html_src_image('../resources/ml-diag.jpg'), width="200", height="200")

        print(doc.getvalue())
        with open(html_file_path, "w") as text_file:
            text_file.write(doc.getvalue())

    def _html_src_image(self, image_path):
        image = PIL.Image.open(image_path)
        data = np.asarray(image)
        image_encoded_str = self._encode_image(data)
        return f'data:image/jpg;base64, {image_encoded_str}'

    def _encode_image(
            self,
            image: Union[np.ndarray, PIL.Image.Image, pathlib.Path, str],
    ) -> str:
        """Encode image to base64 string."""
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                raise RuntimeError(
                    f'image.dtype is {image.dtype}, but it should be uint8.'
                )
            if not (image.ndim == 2 or image.ndim == 3):
                raise RuntimeError(
                    f'image.ndim is {image.ndim}, but it should be 2 or 3.'
                )
            buff = BytesIO()
            PIL.Image.fromarray(image).save(buff, format='PNG')
            encoded = base64.b64encode(buff.getvalue())
        elif isinstance(image, PIL.Image.Image):
            buff = BytesIO()
            image.save(buff, format='PNG')
            encoded = base64.b64encode(buff.getvalue())
        elif isinstance(image, pathlib.Path):
            encoded = base64.b64encode(open(str(image), 'rb').read())
        elif isinstance(image, str):
            encoded = base64.b64encode(open(image, 'rb').read())
        else:
            raise TypeError(
                f'image is of type {type(image)}, but it should be one of: '
                f'{np.ndarray}, {PIL.Image.Image}, {pathlib.Path} or {str}.'
            )
        image_encoded_str = encoded.decode('utf-8')
        return image_encoded_str

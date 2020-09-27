import base64
import os
import pathlib
import shutil
from io import BytesIO
from typing import List, Dict, Union

import PIL
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objs as go
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

    def _get_result_0(self):
        return self.dict_results_augment[0]

    def _format_float(self, value):
        return "{:.2f}".format(value * 100)

    def out(self):
        """
        terminal output
        :return:
        """
        print("Invariance report ...")
        print("======================")
        print(colored("Method", "yellow"),
              "\t", colored("Metric", "blue"),
              "\t", colored("Result (%)", "grey"),
              "\t", colored("Impact", "grey"))
        print("=======================================")
        dict_result_0 = self._get_result_0()
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
        doc, tag, text, line = Doc().ttl()
        doc.asis('<!DOCTYPE html>')
        with tag('html'):
            doc.stag('link', href="./ml-diag.css", rel="stylesheet", type="text/css", media="screen")
            with tag('body'):
                with tag('center'):
                    doc.stag('img', src=self._html_src_image('../resources/ml-diag.jpg'), width="200", height="200")
                    line("h1", "MLDiag: Machine Learning Diagnostics tool")
                line("h2", "Invariance report")
                with tag('table'):
                    with tag('tr'):
                        line('th', "Method")
                        line('th', "Metric")
                        line('th', "Result (%)")
                        line('th', "Impact")
                    dict_result_0 = self._get_result_0()
                    with tag('tr'):
                        line('td', dict_result_0["Method"])
                        line('td', dict_result_0["Metric"])
                        line('td', self._format_float(dict_result_0["Result"]))
                        line('td', "-")

                    for dict_result in self.dict_results_augment[1:]:
                        diff_result = dict_result["Result"] - dict_result_0["Result"]
                        with tag('tr'):
                            line('td', dict_result["Method"])
                            line('td', dict_result["Metric"])
                            line('td', self._format_float(dict_result["Result"]))
                            if diff_result < 0:
                                line('td', self._format_float(diff_result), klass="ko")
                            else:
                                line('td', self._format_float(diff_result), klass="ok")
                line("h2", "Descriptive report")
                with tag('table'):
                    with tag('tr'):
                        line('th', "Method")
                        line('th', "Metric")
                        line('th', "Result (hist)")
                        # line('th', "Impact")
                    for dict_result in self.dict_results_describe:
                        with tag('tr'):
                            line('td', dict_result["Method"])
                            line('td', dict_result["Metric"])
                            with tag('td'):
                                fig = self._plot_hist(res=[dict_result["Result"][0], dict_result["Result"][1]],
                                                      y_data=["hist_ko", "hist_ok"])
                                doc.asis(self._add_plotly_figure(fig))
                            '''
                            #TODO: check why impact plot is always false
                            with tag('td'):
                                fig = self._plot_hist(res=[dict_result["Result"][2]], y_data = ["hist_diff"], 
                                normalize=True)
                                doc.asis(self._add_plotly_figure(fig))
                            '''

        # print(doc.getvalue())
        with open(html_file_path, "w") as text_file:
            text_file.write(doc.getvalue())
        shutil.copyfile("../resources/ml-diag.css", os.path.join(os.path.split(html_file_path)[0], "ml-diag.css"))

    def _plot_hist(self, res, y_data, normalize=False):
        colors = px.colors.sequential.Plasma * 2

        x_data = []

        def result_to_lists(r):
            lst = list(
                zip(*[list(map(int, x.replace("'", "").replace(")", "").replace("(", "").split("%,"))) for x in r]))
            return list(lst[0]), list(lst[1])

        for res_i in res:
            x, top_labels = result_to_lists(res_i)
            x_data.append(x)

        ll = [0] + top_labels
        top_labels = ["[{} - {}]".format(ll[i], ll[i + 1]) for i in range(len(ll) - 1)]

        fig = go.Figure()

        for i in range(0, len(x_data[0])):
            for xd, yd in zip(x_data, y_data):
                fig.add_trace(go.Bar(
                    x=[xd[i]], y=[yd],
                    orientation='h',
                    marker=dict(
                        color=colors[i],
                        line=dict(color='rgb(248, 248, 249)', width=1)
                    )
                ))

        fig.update_layout(
            xaxis=dict(
                showgrid=False,
                showline=False,
                showticklabels=False,
                zeroline=False,
                domain=[0.15, 1]
            ),
            yaxis=dict(
                showgrid=False,
                showline=False,
                showticklabels=False,
                zeroline=False,
            ),
            barmode='stack',
            paper_bgcolor='rgb(248, 248, 255)',
            plot_bgcolor='rgb(248, 248, 255)',
            margin=dict(l=0, r=0, t=30, b=30),
            showlegend=False,
            autosize=False,
            width=800,
            height=300,
        )

        annotations = []
        font_size = 9
        font_size_label = 6
        for yd, xd in zip(y_data, x_data):
            xdd = [abs(x) for x in xd]
            if normalize:
                s = sum(xdd)
                xdd = [int(100 * x / s) for x in xdd]

            # labeling the y-axis
            annotations.append(dict(xref='paper', yref='y',
                                    x=0.14, y=yd,
                                    xanchor='right',
                                    text=str(yd),
                                    font=dict(family='Arial', size=14,
                                              color='rgb(67, 67, 67)'),
                                    showarrow=False, align='right'))
            # labeling the first percentage of each bar (x_axis)
            annotations.append(dict(xref='x', yref='y',
                                    x=xdd[0] / 2, y=yd,
                                    text=str(xd[0]) + '%',
                                    font=dict(family='Arial', size=font_size,
                                              color='rgb(248, 248, 255)'),
                                    showarrow=False))
            # labeling the first Likert scale (on the top)
            if yd == y_data[-1]:
                annotations.append(dict(xref='x', yref='paper',
                                        x=xdd[0] / 2, y=1.1,
                                        text=top_labels[0],
                                        font=dict(family='Arial', size=font_size_label,
                                                  color='rgb(67, 67, 67)'),
                                        showarrow=False,
                                        textangle=-90))
            space = xdd[0]
            for i in range(1, len(xd)):
                # labeling the rest of percentages for each bar (x_axis)
                annotations.append(dict(xref='x', yref='y',
                                        x=space + (xdd[i] / 2) if xdd[i] >= 2 else space + 1, y=yd,
                                        text=str(xd[i]) + '%',
                                        font=dict(family='Arial', size=font_size,
                                                  color='rgb(248, 248, 255)'),
                                        showarrow=False))
                # labeling the Likert scale
                if yd == y_data[-1]:
                    annotations.append(dict(xref='x', yref='paper',
                                            x=space + (xdd[i] / 2) if xdd[i] >= 2 else space + 1, y=1.1,
                                            text=top_labels[i],
                                            font=dict(family='Arial', size=font_size_label,
                                                      color='rgb(67, 67, 67)'),
                                            showarrow=False,
                                            textangle=-90))
                space += xdd[i]

        fig.update_layout(annotations=annotations)
        return fig

    def _add_plotly_figure(
            self,
            fig: plotly.graph_objs.Figure,
            include_plotlyjs: bool = True,
    ) -> str:
        """Add plotly figure."""
        if not isinstance(fig, plotly.graph_objs.Figure):
            raise TypeError(
                f'fig is of type {type(fig)}, '
                f'but it should be {plotly.graph_objs.Figure}.'
            )
        if include_plotlyjs:
            include_plotlyjs = 'cdn'
        else:
            include_plotlyjs = False

        plotly_figure_html = plotly.io.to_html(
            fig=fig,
            full_html=False,
            include_plotlyjs=include_plotlyjs,
        )

        out = (
            '<div class="plotly-figure">\n'
            f'{plotly_figure_html}\n'
            '</div>\n'
        )
        print(out)
        return out

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

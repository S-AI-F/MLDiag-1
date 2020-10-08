import os

import click
import tensorflow as tf
import yaml

from mldiag.session import DiagSession
from mldiag.wrappers import npy_to_numpy, url_to_service


@click.group()
def diag():
    pass


@diag.command()
@click.option(
    '--service_url', '-u',
    help='the url of the prediction service',
    required=True,
)
@click.option(
    '--config_file', '-c',
    help='the path to a config file',
    required=True,
)
@click.option(
    '--eval_set', '-e',
    help='the path to the eval dataset file',
    required=True,
)
@click.option(
    '--report_path', '-r',
    help='the path to output report directory',
    required=True,
)
@click.option(
    '--json_field', '-j',
    help='the json field  data returned by the webservice',
    required=True,
)
def diagnose(service_url, config_file, eval_set, report_path, json_field):
    with open(config_file) as file:
        custom_config = yaml.load(file, Loader=yaml.FullLoader)

    if not os.path.isdir(report_path):
        raise ValueError("Missing output directory {} for reporting".format(report_path))

    DiagSession(
        config=custom_config,
        eval_set=npy_to_numpy(eval_set),
        service=url_to_service(service_url, json_field),
        metric=tf.keras.metrics.BinaryAccuracy(),
        report_path=report_path,
    ).run()


if __name__ == "__main__":
    diag()

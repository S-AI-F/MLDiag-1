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
)
@click.option(
    '--config_file', '-c',
    help='the path to a config file',
)
@click.option(
    '--eval_set', '-e',
    help='the path to the eval dataset file',
)
def diagnose(service_url, config_file, eval_set):
    with open(config_file) as file:
        custom_config = yaml.load(file, Loader=yaml.FullLoader)

    DiagSession(
        config=custom_config,
        eval_set=npy_to_numpy(eval_set),
        service=url_to_service(service_url),
        metric=tf.keras.metrics.BinaryAccuracy()
    ).run()


if __name__ == "__main__":
    diag()

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, jsonify, request

app = Flask(__name__)
app.secret_key = "super_secret_key"

model = None


def load_model(model_path):
    return tf.keras.models.load_model(
        filepath=r"C:\Users\SHABOUA\ws\tmp\mldiag\model.h5",
        custom_objects={'KerasLayer': hub.KerasLayer})


@app.route('/query', methods=['POST', 'GET'])
def input_predict_text():
    text = request.json["text"]
    out = model.predict(np.array(text))
    return jsonify(results=out.tolist())


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentTypeError
    import os


    def file_path(path):
        print(path)
        if os.path.isfile(path):
            return path
        else:
            raise ArgumentTypeError(f"readable_dir:{path} is not a valid path")


    parser = ArgumentParser()
    parser.add_argument('model_path', type=file_path, help="Model Path")
    args = parser.parse_args()
    model = load_model(args.model_path)

    app.run(host="0.0.0.0", port=8080, debug=False)

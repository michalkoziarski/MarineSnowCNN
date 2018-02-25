import argparse
import json
import logging
import imageio
import numpy as np
import tensorflow as tf

from utils import get_network
from pathlib import Path


def load_model(session, model_name):
    model_path = Path(__file__).parent / 'models' / model_name

    with open(model_path / 'params.json') as f:
        params = json.load(f)

    assert model_path.exists()

    inputs = tf.placeholder(tf.float32)
    network = get_network(inputs, params)
    checkpoint = tf.train.get_checkpoint_state(model_path)
    saver = tf.train.Saver()
    saver.restore(session, checkpoint.model_checkpoint_path)

    return network


def predict(inputs, session=None, network=None, model_name=None):
    assert len(inputs.shape) == 5

    session_passed = session is not None

    if not session_passed:
        session = tf.Session()

    if network is None:
        network = load_model(session, model_name)

    predictions = []

    for i in range(len(inputs)):
        frames = inputs[i].copy()

        prediction = network.outputs.eval(feed_dict={network.inputs: np.array([frames])},
                                          session=session)[0]

        predictions.append(prediction)

    if not session_passed:
        session.close()

    return predictions


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('-input', required=True,
                        help='a directory of input frames that will be used in alphabetical order')
    parser.add_argument('-output', required=True,
                        help='a path for the output image')
    parser.add_argument('-model_name', required=True, default='MarineSnowCNN',
                        help='a name of a trained model')

    args = parser.parse_args()

    if not Path(args.input).exists():
        raise ValueError('Incorrect input path.')

    assert len(list(Path(args.input).iterdir())) == 3

    images = []

    for path in Path(args.input).iterdir():
        logging.info('Loading image from path %s...' % path)

        image = imageio.imread(str(path))

        assert image.dtype == np.uint8

        images.append(np.array(image) / 255)

    logging.info('Running prediction...')

    prediction = predict(np.array([images]), model_name=args.model_name)[0]

    imageio.imwrite(args.output, prediction)

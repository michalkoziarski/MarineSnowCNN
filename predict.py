import argparse
import utils
import json
import logging
import imageio
import numpy as np
import tensorflow as tf

from utils import get_network
from pathlib import Path


def load_model(session):
    with open(Path(__file__).parent / 'params.json') as f:
        params = json.load(f)

    checkpoint_path = Path(__file__).parent / 'model'

    assert checkpoint_path.exists()

    inputs = tf.placeholder(tf.float32)
    network = get_network(inputs, params)
    checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
    saver = tf.train.Saver()
    saver.restore(session, checkpoint.model_checkpoint_path)

    return network


def predict(inputs, session=None, network=None, targets=None):
    assert len(inputs.shape) == 5

    session_passed = session is not None

    if not session_passed:
        session = tf.Session()

    if network is None:
        network = load_model(session)

    predictions = []

    if targets is not None:
        psnr = []

    for i in range(len(inputs)):
        frames = inputs[i].copy()

        prediction = network.outputs.eval(feed_dict={network.inputs: np.array([frames])},
                                          session=session)[0]

        if targets is not None:
            psnr.append(utils.psnr(prediction, targets[i], maximum=1.0))

        predictions.append(prediction)

    if not session_passed:
        session.close()

    if targets is not None:
        return predictions, psnr
    else:
        return predictions


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('-input', required=True,
                        help='a directory of input frames that will be used in alphabetical order')
    parser.add_argument('-output', required=True,
                        help='a path for the output image')

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

    prediction = predict(np.array([images]))[0]

    imageio.imwrite(args.output, prediction)

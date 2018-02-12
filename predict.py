import model
import utils
import json
import numpy as np
import tensorflow as tf

from pathlib import Path


def load_model(session):
    with open(Path(__file__).parent / 'params.json') as f:
        params = json.load(f)

    checkpoint_path = Path(__file__).parent / 'model'

    assert checkpoint_path.exists()

    inputs = tf.placeholder(tf.float32)
    network = model.MarineSnowCNN(inputs, params['kernel_size'], params['n_layers'],
                                  params['n_filters'], params['patch_size'][3])
    checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
    saver = tf.train.Saver()
    saver.restore(session, checkpoint.model_checkpoint_path)

    return network


def predict(inputs, session=None, network=None, targets=None):
    with open(Path(__file__).parent / 'params.json') as f:
        params = json.load(f)

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
                                          session=session)[0, params['patch_size'][0] // 2]

        if targets is not None:
            psnr.append(utils.psnr(prediction, targets[i], maximum=1.0))

        predictions.append(prediction)

    if not session_passed:
        session.close()

    if targets is not None:
        return predictions, psnr
    else:
        return predictions

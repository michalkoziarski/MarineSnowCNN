import argparse
import json
import logging
import imageio
import numpy as np
import tensorflow as tf

from data import ImageDataset
from utils import get_network, median_filter
from pathlib import Path
from tqdm import tqdm


def load_model(session, model_name):
    model_path = Path(__file__).parent / 'models' / model_name
    params_path = Path(__file__).parent / 'params' / ('%s.json' % model_name)

    with open(params_path) as f:
        params = json.load(f)

    assert model_path.exists()

    inputs = tf.placeholder(tf.float32)
    network = get_network(inputs, params)
    checkpoint = tf.train.get_checkpoint_state(model_path)
    saver = tf.train.Saver()
    saver.restore(session, checkpoint.model_checkpoint_path)

    return network


def predict_dataset(dataset, session, network, threshold=0.5):
    predictions = []
    metrics = {'TN': 0, 'TP': 0, 'FN': 0, 'FP': 0}

    for _ in tqdm(range(dataset.length)):
        inputs, ground_truth = dataset.fetch()

        prediction = network.outputs.eval(feed_dict={network.inputs: np.array([inputs])}, session=session)[0]

        if threshold is not None:
            prediction[prediction < threshold] = 0.0
            prediction[prediction >= threshold] = 1.0

        predictions.append(prediction)

        metrics['TN'] += np.sum((prediction == 0.0) & (ground_truth == 0.0))
        metrics['TP'] += np.sum((prediction == 1.0) & (ground_truth == 1.0))
        metrics['FN'] += np.sum((prediction == 0.0) & (ground_truth == 1.0))
        metrics['FP'] += np.sum((prediction == 1.0) & (ground_truth == 0.0))

    metrics['accuracy'] = (metrics['TP'] + metrics['TN']) / (metrics['TP'] + metrics['TN'] + metrics['FP'] + metrics['FN'])
    metrics['precision'] = metrics['TP'] / (metrics['TP'] + metrics['FP'])
    metrics['recall'] = metrics['TP'] / (metrics['TP'] + metrics['FN'])
    metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])

    return np.array(predictions), metrics


def filter_dataset(dataset, kernel_size, session, network, predictions=None, threshold=0.5):
    if predictions is None:
        predictions, _ = predict_dataset(dataset, session, network, threshold)

    inputs = [dataset.fetch()[0] for _ in range(len(predictions))]
    outputs = []

    for prediction, input in tqdm(zip(predictions, inputs), total=len(predictions)):
        outputs.append(median_filter(input, kernel_size, prediction, threshold))

    return outputs


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument('-mode', type=str, choices=['detect', 'filter'], default='detect')
    parser.add_argument('-model_name', type=str, default='MarineSnowCNN')
    parser.add_argument('-output_path', type=str, default='outputs')
    parser.add_argument('-dataset', type=str, default='Zakrzowek-B')

    args = parser.parse_args()

    with tf.Session() as session:
        params_path = Path(__file__).parent / 'params' / ('%s.json' % args.model_name)

        with open(params_path) as f:
            params = json.load(f)

        logging.info('Loading dataset...')

        dataset = ImageDataset([args.dataset], params['temporal_patch_size'])

        logging.info('Restoring model...')

        network = load_model(session, args.model_name)

        logging.info('Running prediction...')

        outputs, _ = predict_dataset(dataset, session, network)

        if args.mode == 'filter':
            logging.info('Filtering images...')

            outputs = filter_dataset(dataset, params['temporal_patch_size'], session, network, outputs)

        logging.info('Saving outputs to "%s"...' % args.output_path)

        Path(args.output_path).mkdir(exist_ok=True)

        for i in range(len(outputs)):
            output = outputs[i]
            path = Path(args.output_path) / ('%d.png' % i)
            imageio.imwrite(str(path), output)

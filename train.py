import argparse
import data
import json
import logging
import numpy as np
import tensorflow as tf

from utils import get_network
from tqdm import tqdm
from pathlib import Path
from predict import predict_dataset


MODELS_PATH = Path(__file__).parent / 'models'
LOGS_PATH = Path(__file__).parent / 'logs'
PARAMS_PATH = Path(__file__).parent / 'params'

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument('-model_name', type=str, default='MarineSnowCNN')
parser.add_argument('-n_3d_layers', type=int, default=20)
parser.add_argument('-n_2d_layers', type=int, default=0)
parser.add_argument('-kernel_size', type=int, default=3)
parser.add_argument('-n_filters', type=int, default=64)
parser.add_argument('-epochs', type=int, default=60)
parser.add_argument('-weight_decay', type=float, default=0.0)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-temporal_patch_size', type=int, default=3)
parser.add_argument('-spatial_patch_size', type=int, default=40)
parser.add_argument('-spatial_stride', type=int, default=20)
parser.add_argument('-learning_rate', type=float, default=0.00001)
parser.add_argument('-train_partitions', type=str, nargs='+', default=['Zakrzowek-A'])
parser.add_argument('-validation_partitions', type=str, nargs='+', default=['Antarctica'])
parser.add_argument('-test_partitions', type=str, nargs='+', default=['Zakrzowek-B'])

params = vars(parser.parse_args())

checkpoint_path = MODELS_PATH / params['model_name']
log_path = LOGS_PATH / params['model_name']
model_path = checkpoint_path / 'model.ckpt'
params_path = PARAMS_PATH / ('%s.json' % params['model_name'])

for path in [PARAMS_PATH, checkpoint_path, log_path]:
    path.mkdir(parents=True, exist_ok=True)

if params_path.exists():
    logging.info('Loading model parameters from %s...' % params_path)

    with open(params_path) as f:
        saved_params = json.load(f)

    if params != saved_params:
        raise ValueError('Saved parameters are different than the ones passed to the trainer.')
else:
    logging.info('Saving model parameters to %s...' % params_path)

    with open(params_path, 'w') as f:
        json.dump(params, f)

logging.info('Loading training dataset...')

train_set = data.PatchDataset(params['train_partitions'], params['batch_size'], params['temporal_patch_size'],
                              params['spatial_patch_size'], params['spatial_stride'])

logging.info('Loading validation dataset...')

validation_set = data.ImageDataset(params['validation_partitions'], params['temporal_patch_size'])

logging.info('Loading test dataset...')

test_set = data.ImageDataset(params['test_partitions'], params['temporal_patch_size'])

inputs = tf.placeholder(tf.float32)
ground_truth = tf.placeholder(tf.float32)
global_step = tf.Variable(0, trainable=False, name='global_step')
network = get_network(inputs, params)
base_loss = tf.losses.mean_squared_error(network.outputs, ground_truth)
weight_loss = params['weight_decay'] * tf.reduce_sum(tf.stack([tf.nn.l2_loss(weight) for weight in network.weights]))
loss = base_loss + weight_loss

accuracy = tf.placeholder(tf.float32, shape=[])
precision = tf.placeholder(tf.float32, shape=[])
recall = tf.placeholder(tf.float32, shape=[])
f1_score = tf.placeholder(tf.float32, shape=[])

tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('precision', precision)
tf.summary.scalar('recall', recall)
tf.summary.scalar('f1_score', f1_score)
tf.summary.image('inputs', inputs[:, params['temporal_patch_size'] // 2])
tf.summary.image('outputs', network.outputs)

for i in range(len(network.weights)):
    tf.summary.histogram('weights/layer_%d' % (i + 1), network.weights[i])
    tf.summary.histogram('biases/layer_%d' % (i + 1), network.biases[i])

summary_step = tf.summary.merge_all()
saver = tf.train.Saver(max_to_keep=None)

optimizer = tf.train.AdamOptimizer(params['learning_rate'])
train_step = optimizer.minimize(loss, global_step=global_step)

summary_writer = tf.summary.FileWriter(str(log_path))

with tf.Session() as session:
    checkpoint = tf.train.get_checkpoint_state(str(checkpoint_path))

    if checkpoint and checkpoint.model_checkpoint_path:
        logging.info('Restoring model...')

        session.run(tf.global_variables_initializer())
        saver.restore(session, checkpoint.model_checkpoint_path)
    else:
        logging.info('Initializing new model...')

        session.run(tf.global_variables_initializer())

    logging.info('Training model...')

    batches_processed = tf.train.global_step(session, global_step)
    epochs_processed = int(batches_processed * params['batch_size'] / train_set.length)
    batches_per_epoch = int(np.ceil(train_set.length / train_set.batch_size))

    for epoch in range(epochs_processed, params['epochs']):
        logging.info('Shuffling dataset...')

        train_set.shuffle()

        logging.info('Processing epoch #%d...' % (epoch + 1))

        for batch in tqdm(range(0, batches_per_epoch)):
            x, y = train_set.batch()

            feed_dict = {inputs: x, ground_truth: y}

            session.run([train_step], feed_dict=feed_dict)

        logging.info('Evaluating model on validation set...')

        _, metrics = predict_dataset(validation_set, session, network)

        logging.info('Observed accuracy = %.4f, precision = %.4f, recall = %.4f, F1 score = %.4f' %
                     (metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']))

        summary = session.run([summary_step], feed_dict={
            accuracy: metrics['accuracy'],
            precision: metrics['precision'],
            recall: metrics['recall'],
            f1_score: metrics['f1_score']
        })

        saver.save(session, str(model_path), global_step=(epoch + 1))
        summary_writer.add_summary(summary, epoch + 1)

    logging.info('Training complete.')
    logging.info('Evaluating model on test set...')

    _, metrics = predict_dataset(validation_set, session, network)

    logging.info('Observed accuracy = %.4f, precision = %.4f, recall = %.4f, F1 score = %.4f' %
                 (metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']))

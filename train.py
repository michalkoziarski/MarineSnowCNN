import argparse
import data
import json
import logging
import numpy as np
import tensorflow as tf

from utils import get_network
from tqdm import tqdm
from pathlib import Path


MODELS_PATH = Path(__file__).parent / 'models'
LOGS_PATH = Path(__file__).parent / 'logs'

logging.basicConfig(level=logging.INFO)

with open(Path(__file__).parent / 'params.json') as f:
    default_params = json.load(f)

parser = argparse.ArgumentParser()

for k, v in default_params.items():
    parser.add_argument('-%s' % k, type=type(v), default=v)

params = vars(parser.parse_args())

checkpoint_path = MODELS_PATH / params['model_name']
log_path = LOGS_PATH / params['model_name']
model_path = checkpoint_path / 'model.ckpt'
params_path = checkpoint_path / 'params.json'

for path in [checkpoint_path, log_path]:
    path.mkdir(parents=True, exist_ok=True)

if params_path.exists():
    logging.info('Loading model parameters from %s...' % params_path)

    with open(params_path) as f:
        saved_params = json.load(f)

    if params != saved_params:
        logging.warning('Detected differences in parameters loaded from %s. Using loaded parameters.' % params_path)

        params_path = saved_params
else:
    logging.info('Saving model parameters to %s...' % params_path)

    with open(params_path, 'w') as f:
        json.dump(params, f)

logging.info('Loading training dataset...')

train_set = data.AnnotatedDataset(params['train_partitions'], params['batch_size'], params['patch_size'],
                                  params['stride'], params['annotation_type'])

inputs = tf.placeholder(tf.float32)
ground_truth = tf.placeholder(tf.float32)
global_step = tf.Variable(0, trainable=False, name='global_step')
network = get_network(inputs, params)
base_loss = tf.losses.mean_squared_error(network.outputs, ground_truth)
weight_loss = params['weight_decay'] * tf.reduce_sum(tf.stack([tf.nn.l2_loss(weight) for weight in network.weights]))
loss = base_loss + weight_loss

tf.summary.scalar('base_loss', base_loss)
tf.summary.scalar('weight_loss', weight_loss)
tf.summary.scalar('total_loss', loss)
tf.summary.image('ground_truth', ground_truth)
tf.summary.image('inputs', inputs[:, params['patch_size'][0] // 2])
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

            if batch < batches_per_epoch - 1:
                session.run([train_step], feed_dict=feed_dict)
            else:
                _, summary = session.run([train_step, summary_step], feed_dict=feed_dict)

                saver.save(session, str(model_path), global_step=(epoch + 1))
                summary_writer.add_summary(summary, epoch)

    logging.info('Training complete.')

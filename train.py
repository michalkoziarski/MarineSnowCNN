import data
import model
import json
import logging
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from pathlib import Path


logging.basicConfig(level=logging.INFO)

with open(Path(__file__).parent / 'params.json') as f:
    params = json.load(f)

logging.info('Loading training dataset...')

train_set = data.AnnotatedDataset(params['train_partitions'], params['batch_size'],
                                  params['patch_size'], params['stride'])

inputs = tf.placeholder(tf.float32)
ground_truth = tf.placeholder(tf.float32)
global_step = tf.Variable(0, trainable=False, name='global_step')
network = model.MarineSnowCNN(inputs, params['kernel_size'], params['n_layers'], params['n_filters'])

base_loss = tf.losses.mean_squared_error(network.outputs[:, params['patch_size'][0] // 2], ground_truth)
weight_loss = params['weight_decay'] * tf.reduce_sum(tf.stack([tf.nn.l2_loss(weight) for weight in network.weights]))
loss = base_loss + weight_loss

tf.summary.scalar('base_loss', base_loss)
tf.summary.scalar('weight_loss', weight_loss)
tf.summary.scalar('total_loss', loss)
tf.summary.image('inputs', inputs)
tf.summary.image('ground_truth', ground_truth)
tf.summary.image('outputs', network.outputs[:, params['patch_size'][0] // 2])
tf.summary.image('residual', network.residual[:, params['patch_size'][0] // 2])

for i in range(len(network.weights)):
    tf.summary.histogram('weights/layer_%d' % (i + 1), network.weights[i])
    tf.summary.histogram('biases/layer_%d' % (i + 1), network.biases[i])

summary_step = tf.summary.merge_all()
saver = tf.train.Saver(max_to_keep=None)

optimizer = tf.train.AdamOptimizer(params['learning_rate'])
train_step = optimizer.minimize(loss, global_step=global_step)

checkpoint_path = Path(__file__).parent / 'model'
model_path = checkpoint_path / 'model.ckpt'
log_path = Path(__file__).parent / 'log'

summary_writer = tf.summary.FileWriter(log_path)

for path in [checkpoint_path, log_path]:
    if not path.exists():
        path.mkdir()

with tf.Session() as session:
    checkpoint = tf.train.get_checkpoint_state(checkpoint_path)

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

                saver.save(session, model_path)
                summary_writer.add_summary(summary, epoch)

    logging.info('Training complete.')

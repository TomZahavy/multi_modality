from datetime import datetime
from tensorflow.python.platform import gfile
from conv_net import simple_conv_word_embedding
import tensorflow as tf
import time
import numpy as np
import os
from conv_net.simple_conv_word_embedding import generate_sentence_and_label_batch, get_predictions, get_accuracy, NUM_EPOCHS_PER_DECAY, NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN, \
    INITIAL_LEARNING_RATE, LEARNING_RATE_DECAY_FACTOR

class data_source():
    def __init__(self,name,size,filters):
        self.name = name
        self.size = size
        self.filters = filters

def train(max_steps, train_dir, restore_from_ckpt, checkpoint_dir, train_batch_size=5000, validation_batch_size=100, train_filenames=[], validation_filenames=[],
          network_spec=simple_conv_word_embedding, log_device_placement=True, size_embedding=100, size=40, num_classes=None, vocab_size=None, min_queue_examples=10, penalty=.0):

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-0.1, 0.1)

        for f in train_filenames:
            if not gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)

        for f in validation_filenames:
            if not gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)

        # Create a queue that produces the filenames to read.
        train_filename_queue = tf.train.string_input_producer(train_filenames)
        validation_filename_queue = tf.train.string_input_producer(validation_filenames)

        data_sources = []
        data_sources.append(data_source('title',40,[3, 4, 5]))
        #data_sources.append(data_source('description',300,[3, 4, 5]))

        with tf.variable_scope("model", reuse=None, initializer=initializer):
            sentence, label, extra = network_spec.read_and_decode_sentence(train_filename_queue,data_sources)
            sentences, labels, extras = generate_sentence_and_label_batch(sentence, label, extra, min_queue_examples=min_queue_examples, batch_size=train_batch_size)

            # Build a Graph that computes the logits predictions from the inference model.
            logits = network_spec.nce_inference(sentences, extras, data_sources, vocab_size, size_embedding, num_classes, penalty=penalty,calc_acc=True)

            # predictions
            predictions = get_predictions(logits)
            accuracy, avg_accuracy = get_accuracy(predictions, labels, summary_postfix='train')

        with tf.variable_scope("model", reuse=True, initializer=initializer):
            # Get images and labels for CIFAR-10.
            sentence_val, label_val, extra_val = network_spec.read_and_decode_sentence(validation_filename_queue,data_sources)
            sentences_val, labels_val, extras_val = generate_sentence_and_label_batch(sentence_val, label_val, extra_val, min_queue_examples=min_queue_examples,
                                                                                      batch_size=validation_batch_size)

            # Build a Graph that computes the logits predictions from the inference model.
            logits_val = network_spec.nce_inference(sentences_val, extras_val, data_sources, vocab_size, size_embedding, num_classes, penalty=penalty, training=False,calc_acc=True)

            # predictions
            predictions_val = get_predictions(logits_val)
            accuracy_val, avg_accuracy_val = get_accuracy(predictions_val, labels_val, summary_postfix='validation')

        # Calculate loss.
        loss = network_spec.nce_inference(sentences, extras, data_sources, vocab_size, size_embedding, num_classes,
                                       penalty=penalty,labels=labels)

        global_step = tf.Variable(0, trainable=False)
        # Build a Graph that trains the model with one batch of examples and updates the model parameters.
        train_op = network_spec.train(loss, global_step, batch_size=train_batch_size)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Initialize model/embedding first before initializing anything else.
        print [v.name for v in tf.all_variables()]
        vars = [v for v in tf.all_variables() if v.name == 'model/embedding:0']
        print 'Variables: ', len(vars)
        init_model_embedding = tf.initialize_variables(vars)

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=log_device_placement))
        sess.run(init_model_embedding)
        sess.run(init)

        if restore_from_ckpt:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print 'Restored from checkpoint'
            else:
                print 'No checkpoint found'

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.train.SummaryWriter(train_dir,
                                                graph_def=sess.graph_def)

        for step in xrange(max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if (step + 1) % 10 == 0:
                num_examples_per_step = train_batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value,
                                     examples_per_sec, sec_per_batch))

            if (step + 1) % 100 == 0:
                summary_str, avg_acc, avg_acc_val = sess.run([summary_op, avg_accuracy, avg_accuracy_val])
                summary_writer.add_summary(summary_str, step)
                # format_str = ('%s: step %d, validation_accuracy = %.2f')
                # print (format_str % (datetime.now(), step, avg_acc_val))

            # Save the model checkpoint periodically and evaluate the model.
            if (step + 1) % 1000 == 0 or (step + 1) == max_steps:
                checkpoint_path = os.path.join(train_dir, 'model_{}.ckpt'.format(step))
                saver.save(sess, checkpoint_path, global_step=step)

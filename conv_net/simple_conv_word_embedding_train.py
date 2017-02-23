from datetime import datetime
from tensorflow.python.platform import gfile
from conv_net import simple_conv_word_embedding
import tensorflow as tf
import time
import numpy as np
import os
import json
from conv_net.simple_conv_word_embedding import generate_sentence_and_label_batch, get_accuracy,log_accuracy,generate_sentence_and_label_batch_noshuffle
from parsing.parse import parse_sentence_padd_batch
from util.vgg import vgg16


def train(max_steps, train_dir, restore_from_ckpt, checkpoint_dir, train_filenames=[], validation_filenames=[],
          network_spec=simple_conv_word_embedding, log_device_placement=True, size_embedding=100, vocab_size=None, min_queue_examples=10,
          metadata=[],dictionary=None):

    train_batch_size = metadata.train_batch_size
    validation_batch_size = metadata.validation_batch_size

    with tf.Graph().as_default():
        imgs = tf.placeholder(tf.float32, [None, metadata.imsize, metadata.imsize, 3])  #
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



        with tf.variable_scope("model",reuse=None,initializer=initializer):

            single_data_source, label = network_spec.read_and_decode_sentence(train_filename_queue,metadata)
            train_data_batch = generate_sentence_and_label_batch(single_data_source, label, min_queue_examples=min_queue_examples, batch_size=train_batch_size)

            text = []
            extras = []
            image_features = []
            for ds in metadata.data_sources:
                if ds.name == 'image' or ds.name == 'image_features' or ds.name == 'image_logits':
                    image_features = [tf.placeholder(tf.float32, shape=(train_batch_size, ds.size))]
                else:
                    text.append(tf.placeholder(tf.int32,shape=(train_batch_size,ds.size)))
                    extras.append(tf.placeholder(tf.int32,shape=(train_batch_size,ds.size,3)))
            labels_data_train = [tf.placeholder(tf.int32,shape=(train_batch_size,metadata.num_classes))]

            # Build a Graph that computes the logits predictions from the inference model.
            logits,_,h_pool = network_spec.inference2(text, extras, vocab_size, size_embedding,image_features=image_features, metadata=metadata)
            accuracy = get_accuracy(logits, labels_data_train[0])


        with tf.variable_scope("model",reuse=True,initializer=initializer):
            # Get images and labels for CIFAR-10.
            single_data_source_val, label_val = network_spec.read_and_decode_sentence(validation_filename_queue,metadata)
            validation_data_batch = generate_sentence_and_label_batch(single_data_source_val, label_val, min_queue_examples=min_queue_examples,
                                                                                      batch_size=validation_batch_size)
            image_features_val = []
            text_val = []
            extras_val = []
            for ds in metadata.data_sources:
                if ds.name == 'image' or ds.name == 'image_features' or ds.name == 'image_logits':
                    image_features_val = [tf.placeholder(tf.float32, shape=(validation_batch_size, ds.size))]
                else:
                    text_val.append(tf.placeholder(tf.int32, shape=(validation_batch_size, ds.size)))
                    extras_val.append(tf.placeholder(tf.int32, shape=(validation_batch_size, ds.size, 3)))

            labels_data_val = [tf.placeholder(tf.int32,shape=(validation_batch_size,metadata.num_classes))]
            # Build a Graph that computes the logits predictions from the inference model.
            logits_val,_,_ = network_spec.inference2(text_val, extras_val, vocab_size, size_embedding, metadata=metadata, training=False,image_features=image_features_val)
            accuracy_val = get_accuracy(logits_val, labels_data_val[0])


        # Calculate loss.
        accuracy_value = tf.placeholder(tf.float32,shape=1)
        accuracy_value_val = tf.placeholder(tf.float32,shape=1)
        accuracy_log,accuracy_avg_log = log_accuracy(accuracy_value, summary_postfix='train')
        accuracy_val_log,accuracy_avg_val_log = log_accuracy(accuracy_value_val, summary_postfix='validation')

        loss = network_spec.loss(logits, labels_data_train[0], metadata)

        global_step = tf.Variable(0, trainable=False)
        # Build a Graph that trains the model with one batch of examples and updates the model parameters.
        train_op = network_spec.train(loss, global_step, metadata=metadata)

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

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement))
        sess.run(init_model_embedding)
        sess.run(init)

        if metadata.image_flag or metadata.logits_flag:
            vgg = vgg16(imgs)
            if not hasattr(metadata,'vgg_checkpoint'):
                vgg.load_weights(metadata.vgg_weights, sess)

        load_mix_network(metadata, restore_from_ckpt, checkpoint_dir, sess)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.train.SummaryWriter(train_dir,graph_def=sess.graph_def)
        for step in xrange(max_steps):
            start_time = time.time()
            train_data = sess.run(train_data_batch)
            # import matplotlib.pyplot as plt
            # plt.imshow(train_data[0][2].astype(int))

            train_data_concat,im = decode_data_stream(train_data,dictionary,train_batch_size,metadata)
            if metadata.imagefeatures_flag:
                train_data_concat.append(im)
            elif metadata.image_flag:
                train_data_concat.append(sess.run(vgg.fc2, feed_dict={vgg.imgs: im}))
            elif metadata.logits_flag:
                train_data_concat.append(sess.run(vgg.logits, feed_dict={vgg.imgs: im}))

            _, loss_value = sess.run([train_op, loss],
                                     feed_dict={i: d for i, d in zip(text+extras+labels_data_train+image_features, train_data_concat)})

            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if (step + 1) % 10 == 0:
                num_examples_per_step = train_batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f ''sec/batch)')
                print (format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

            if (step + 1) % 100 == 1:
                # Evaluate the model, compute accuracy
                acc_train_list = []
                acc_val_list   = []
                # compute multiple batches of acc to get more robust results
                for i in xrange(metadata.validation_iters):
                    val_data = sess.run(validation_data_batch)
                    train_data = sess.run(train_data_batch)


                    train_data_concat,im_train = decode_data_stream(train_data, dictionary, train_batch_size, metadata)
                    val_data_concat,im_val = decode_data_stream(val_data, dictionary, validation_batch_size, metadata)
                    if metadata.imagefeatures_flag:
                        train_data_concat.append(im_train)
                        val_data_concat.append(im_val)
                    elif metadata.image_flag:
                        val_data_concat.append(sess.run(vgg.fc2, feed_dict={vgg.imgs: im_val}))
                        train_data_concat.append(sess.run(vgg.fc2, feed_dict={vgg.imgs: im_train}))
                    elif metadata.logits_flag:
                        val_data_concat.append(sess.run(vgg.logits, feed_dict={vgg.imgs: im_val}))
                        train_data_concat.append(sess.run(vgg.logits, feed_dict={vgg.imgs: im_train}))
                    acc, acc_val = sess.run([accuracy, accuracy_val],
                                                                  feed_dict={i: d for i, d in zip(text + extras +
                                                                  labels_data_train+image_features+text_val + extras_val +
                                                                  labels_data_val+image_features_val, train_data_concat+val_data_concat)})
                    acc_train_list.append(acc)
                    acc_val_list.append(acc_val)
                acc_avg_train = np.zeros(shape=1)
                acc_avg_train[0] = sum(acc_train_list) / len(acc_train_list)
                acc_avg_val = np.zeros(shape=1)
                acc_avg_val[0] = sum(acc_val_list)/len(acc_val_list)
                summary_str = sess.run([summary_op,accuracy_log,accuracy_avg_log,accuracy_val_log,accuracy_avg_val_log],
                              feed_dict={i: d for i,d in zip(text + extras + labels_data_train+image_features+
                              text_val + extras_val + labels_data_val+image_features_val+
                              [accuracy_value]+[accuracy_value_val],
                              train_data_concat+val_data_concat+ [acc_avg_train]+[acc_avg_val])})
                summary_writer.add_summary(summary_str[0], step)


            if (step + 1) % 1000 == 10 or (step + 1) == max_steps:

                # save memory and run time data
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

                val_data = sess.run(validation_data_batch)

                run_metadata_a = tf.RunMetadata()
                train_data = sess.run(train_data_batch,options = run_options,run_metadata = run_metadata_a)
                summary_writer.add_run_metadata(run_metadata_a, 'a_step%d' % step)

                # train_data_concat, im_train = decode_data_stream(train_data, dictionary, train_batch_size, metadata)
                # val_data_concat, im_val = decode_data_stream(val_data, dictionary, validation_batch_size, metadata)
                # if metadata.imagefeatures_flag:
                #     train_data_concat.append(im_train)
                #     val_data_concat.append(im_val)
                # elif metadata.image_flag:
                #     val_data_concat.append(sess.run(vgg.fc2, feed_dict={vgg.imgs: im_val}))
                #     run_metadata_b = tf.RunMetadata()
                #     train_data_concat.append(sess.run(vgg.fc2, feed_dict={vgg.imgs: im_train},options = run_options,run_metadata = run_metadata_b))
                #     summary_writer.add_run_metadata(run_metadata_b, 'b_step%d' % step)
                #
                # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                # run_metadata_c = tf.RunMetadata()
                # sess.run([accuracy, accuracy_val],feed_dict={i: d for i, d in zip(text + extras +
                #           labels_data_train+image_features+text_val + extras_val + labels_data_val+image_features_val,
                #           train_data_concat+val_data_concat)}, options = run_options,run_metadata = run_metadata_c)
                # summary_writer.add_run_metadata(run_metadata_c, 'c_step%d' % step)

                # Save the model checkpoint periodically and evaluate the model.
                checkpoint_path = os.path.join(train_dir, 'model_{}.ckpt'.format(step))
                saver.save(sess, checkpoint_path, global_step=step)


def load_mix_network(metadata,restore_from_ckpt,checkpoint_dir,sess):

    #load text part
    vars = [v for v in tf.all_variables() if (('model' in v.name) and ('softmax') not in v.name and ('image_fully_connected') not in v.name and ('policy') not in v.name)]
    vars2 = [v for v in tf.all_variables() if ('title_softmax' in v.name)and ('policy') not in v.name]

    if len(vars)>0:
        saver = tf.train.Saver(vars)
        if restore_from_ckpt:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, checkpoint_dir+'/'+ckpt.model_checkpoint_path.split('/')[-1])
                if metadata.trainpolicy:
                    saver = tf.train.Saver({"model/softmax_linear/softmax_linear_weights": vars2[0]})
                    saver.restore(sess, checkpoint_dir + '/' + ckpt.model_checkpoint_path.split('/')[-1])
                    saver = tf.train.Saver({"model/softmax_linear/softmax_linear_biases": vars2[1]})
                    saver.restore(sess, checkpoint_dir + '/' + ckpt.model_checkpoint_path.split('/')[-1])
                print 'Restored from checkpoint'
            else:
                print 'No checkpoint found'


    #load image part
    vars = [v for v in tf.all_variables() if ((('conv' in v.name) or ('fc' in v.name) or ('preprocess' in v.name)) and ('Adam'not in v.name and 'model' not in v.name and ('policy') not in v.name))]
    vars2 = [v for v in tf.all_variables() if ((('softmax' in v.name) ) and ('policy' not in v.name))]
    if len(vars)>0:
        saver = tf.train.Saver(vars)
        if restore_from_ckpt:
            ckpt = tf.train.get_checkpoint_state(metadata.vgg_checkpoint)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, metadata.vgg_checkpoint + '/' + ckpt.model_checkpoint_path.split('/')[-1])
                saver = tf.train.Saver({"model/softmax_linear/softmax_linear_weights": vars2[2]})
                saver.restore(sess, metadata.vgg_checkpoint + '/' + ckpt.model_checkpoint_path.split('/')[-1])
                saver = tf.train.Saver({"model/softmax_linear/softmax_linear_biases": vars2[3]})
                saver.restore(sess, metadata.vgg_checkpoint + '/' + ckpt.model_checkpoint_path.split('/')[-1])
                print 'VGG Restored from checkpoint'
            else:
                print 'No VGG checkpoint found'

def decode_data_stream(data,dictionary,batch_size,metadata):
    # the function assumes data is a list, where 0 corresponds to labels and 1-end corresponds to raw text
    labels = calc_labels(data[-1],batch_size,metadata)
    textdata = []
    extrasdata =[]
    images = []

    ## change code here. now depends on: label: text: image

    for data_source,curr_data in zip(metadata.data_sources,data):
        if data_source.name =='image' or data_source.name =='image_logits':
            images = curr_data
        elif data_source.name =='image_features':
            images = np.reshape(curr_data.values,(batch_size,data_source.size))
        else:
            currtextdata, currextrasdata = parse_sentence_padd_batch(dictionary,curr_data , batch_size, size=data_source.size)
            textdata.append(currtextdata)
            extrasdata.append(currextrasdata)


    outdata = textdata+extrasdata+[labels]
    return outdata,images

def calc_labels(label_strings,train_batch_size,metadata):
    labels = np.zeros(shape=(train_batch_size, metadata.num_classes), dtype=int)
    for prod_id, label_string in enumerate(label_strings[1]):
        for tag in json.loads(label_string):
            if tag['decision'] == 'is':
                if tag['value_id'] in metadata.labelmap:
                    labels[prod_id, metadata.labelmap[tag['value_id']]] = 1
                else:
                    print 'tag ' + tag['value_id'] + ' not found'
    return labels

def test_label_from_ckpt(checkpoint_dir, val_size=5000,validation_filenames=[],
                         network_spec=simple_conv_word_embedding, log_device_placement=True, size_embedding=100,
                         vocab_size=None, min_queue_examples=10,metadata=[], dictionary=None,fromfile=False):
    import pickle
    from util.bh_tsne import *
    validation_batch_size = 16
    suffix = str(val_size)+'image_90'
    if fromfile:
        print('load data')
        with open('/Users/tzahavy/git/code/tensorflow-classifiers/tsne_dat_'+suffix+'.pkl', 'rb') as pickle_load:
            Y = pickle.load(pickle_load)
        with open('/Users/tzahavy/git/code/tensorflow-classifiers/tsne_acc_'+suffix+'.pkl', 'rb') as pickle_load:
            acc = pickle.load(pickle_load)
        with open('/Users/tzahavy/git/code/tensorflow-classifiers/tsne_labels_' + suffix + '.pkl','rb') as pickle_load:
            pred = pickle.load(pickle_load)
        print('load data done')

    else:
        with tf.Graph().as_default():
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
            for f in validation_filenames:
                if not gfile.Exists(f):
                    raise ValueError('Failed to find file: ' + f)

            # Create a queue that produces the filenames to read.
            validation_filename_queue = tf.train.string_input_producer(validation_filenames,shuffle=False)
            with tf.variable_scope("model", reuse=None, initializer=initializer):

                single_data_source, label = network_spec.read_and_decode_sentence(validation_filename_queue, metadata)
                validation_data_batch = generate_sentence_and_label_batch_noshuffle(single_data_source, label,
                                                                     min_queue_examples=min_queue_examples,
                                                                     batch_size=validation_batch_size)
                imgs = validation_data_batch[0]
                text_val = []
                extras_val = []
                image_features_val = []
                for ds in metadata.data_sources:
                    if ds.name == 'image' or ds.name == 'image_features':
                        image_features_val = [tf.placeholder(tf.float32, shape=(validation_batch_size, ds.size))]
                    else:
                        text_val.append(tf.placeholder(tf.int32, shape=(validation_batch_size, ds.size)))
                        extras_val.append(tf.placeholder(tf.int32, shape=(validation_batch_size, ds.size, 3)))
                labels_data_val = [tf.placeholder(tf.int32, shape=(validation_batch_size, metadata.num_classes))]

                # Build a Graph that computes the logits predictions from the inference model.
                logits, activations, h_pool = network_spec.inference(text_val, extras_val, vocab_size, size_embedding,
                                                           image_features=image_features_val, metadata=metadata)

            # Build an initialization operation to run below.
            init = tf.initialize_all_variables()
            # Initialize model/embedding first before initializing anything else.
            print [v.name for v in tf.all_variables()]
            vars = [v for v in tf.all_variables() if v.name == 'model/embedding:0']
            print 'Variables: ', len(vars)
            init_model_embedding = tf.initialize_variables(vars)
            # Start running operations on the Graph.
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement))
            sess.run(init_model_embedding)
            sess.run(init)

            if metadata.image_flag:
                vgg = vgg16(imgs)
            saver = tf.train.Saver(tf.all_variables())
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, checkpoint_dir+'/'+ckpt.model_checkpoint_path.split('/')[-1])
                print 'Restored from checkpoint'


            # Start the queue runners.
            tf.train.start_queue_runners(sess=sess)
            print('infer data')

            for i in xrange(val_size/validation_batch_size):
                print(str(i))
                val_data = sess.run(validation_data_batch)
                print val_data[1][1][2]

                val_data_concat,im = decode_data_stream(val_data, dictionary, validation_batch_size, metadata)
                label_ind = -1
                if metadata.imagefeatures_flag:
                    val_data_concat.append(im)
                    label_ind = -2

                elif metadata.image_flag:
                    val_data_concat.append(sess.run(vgg.fc2, feed_dict={vgg.imgs: im}))
                    label_ind=-2
                logits_curr,X_curr = sess.run([logits,activations],feed_dict={i: d for i, d in zip(text_val + extras_val + labels_data_val+image_features_val,
                                                                                             val_data_concat)})
                if i==0:
                    preds = logits_curr
                    X = X_curr
                    labels = val_data_concat[label_ind]
                else:
                    preds = np.concatenate((preds, logits_curr),axis=0)
                    X = np.concatenate((X,X_curr),axis=0)
                    labels = np.concatenate((labels,val_data_concat[label_ind]),axis=0)

        pred =  np.argmax(preds,1)
        acc = (pred==np.argmax(preds*labels,1)).astype('float')
        Y = np.zeros(shape=(X.shape[0], 2))
        print('infer data done')
        print('run tsne')

        for i, result in enumerate(bh_tsne(X, no_dims=2, perplexity=90, theta=0.5)):
            Y[i, :] = result
        with open('tsne_dat_'+suffix+'.pkl', 'wb') as pickle_file:
            pickle.dump(Y, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        with open('tsne_acc_'+suffix+'.pkl', 'wb') as pickle_file:
            pickle.dump(acc, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        with open('tsne_labels_'+suffix+'.pkl', 'wb') as pickle_file:
            pickle.dump(pred, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        with open('tsne_preds_' + suffix + '.pkl', 'wb') as pickle_file:
            pickle.dump(softmax(preds), pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        print('run tsne done')

    tsne_plot(Y,acc,pred)

def tsne_plot(Y,acc,pred,acc2=[]):
    from matplotlib import pyplot as plt
    if len(acc2) == 0:
        acc2=acc
    plt.close("all")

    plt.figure()
    plt.scatter(Y[:, 0], Y[:, 1], s=8, c=acc, lw=0)

    plt.figure()
    plt.scatter(Y[:, 0], Y[:, 1], s=8, c=pred, lw=0)

    plt.figure()
    indices = np.nonzero(np.logical_and(acc == 1,acc2==1))[0]
    a_text = 'text:True,image:True: ' + str(float(len(indices))/len(acc))
    a = plt.scatter(Y[indices, 0], Y[indices, 1], s=8,c='red', edgecolors='none')
    indices = np.nonzero(np.logical_and(acc == 1 , acc2==0))[0]
    b_text =  'text:True,image:False: ' + str(float(len(indices)) / len(acc))
    b = plt.scatter(Y[indices, 0], Y[indices, 1], s=8,c='blue', edgecolors='none')
    indices = np.nonzero(np.logical_and(acc == 0 , acc2 == 1))[0]
    c_text =  'text:False,image:True: ' + str(float(len(indices)) / len(acc))
    c = plt.scatter(Y[indices, 0], Y[indices, 1], s=8, c='green', edgecolors='none')
    indices = np.nonzero(np.logical_and(acc == 0 , acc2 == 0))[0]
    d_text = 'text:False,image:False: ' + str(float(len(indices)) / len(acc))
    d = plt.scatter(Y[indices, 0], Y[indices, 1], s=8, c='black', edgecolors='none')
    plt.legend((a, b, c, d),
               (a_text,b_text,c_text,d_text))
    plt.show(block=True)




def write_features(train_dir,output_dir, train_filenames=[], validation_filenames=[], network_spec=simple_conv_word_embedding,
                   log_device_placement=True, min_queue_examples=10, metadata=[], dictionary=None):


    os.makedirs(output_dir)
    output_training_file_template = 'features_train_{}.bin'
    output_validation_file_template='features_validation_{}.bin'
    train_batch_size = 100
    validation_batch_size = 100

    with tf.Graph().as_default():
        imgs = tf.placeholder(tf.float32, [None, metadata.imsize, metadata.imsize, 3])  #

        for f in train_filenames:
            if not gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)

        for f in validation_filenames:
            if not gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)

        # Create a queue that produces the filenames to read.
        train_filename_queue = tf.train.string_input_producer(train_filenames)
        validation_filename_queue = tf.train.string_input_producer(validation_filenames)

        with tf.variable_scope("model", reuse=None):

            single_data_source, label = network_spec.read_and_decode_sentence(train_filename_queue, metadata)
            train_data_batch = generate_sentence_and_label_batch_noshuffle(single_data_source, label,
                                                                 min_queue_examples=min_queue_examples,
                                                                 batch_size=train_batch_size)

        with tf.variable_scope("model", reuse=True):
            single_data_source_val, label_val = network_spec.read_and_decode_sentence(validation_filename_queue, metadata)
            validation_data_batch = generate_sentence_and_label_batch_noshuffle(single_data_source_val, label_val,
                                                                      min_queue_examples=min_queue_examples,
                                                                      batch_size=validation_batch_size)


        sess = tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement))
        if metadata.image_flag:
            vgg = vgg16(imgs)
            if metadata.vgg_checkpoint is None:
                vgg.load_weights(metadata.vgg_weights, sess)

        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(metadata.vgg_checkpoint)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, metadata.vgg_checkpoint + '/' + ckpt.model_checkpoint_path.split('/')[-1])
            print 'Restored from checkpoint'

        tf.train.start_queue_runners(sess=sess)


    # assume each file is of size 1000 and reading with batch = 100
        batch_examples_train = []
        for step in xrange(len(train_filenames)*10):
            print step
            train_data = sess.run(train_data_batch)
            _, im = decode_data_stream(train_data, dictionary, 100, metadata)
            vgg_features = sess.run(vgg.fc2, feed_dict={vgg.imgs: im})
            batch_examples_train += write_example_batch(train_data,vgg_features, train_batch_size)
            if step % 10 == 9:
                write_batch(batch_examples_train, os.path.join(output_dir, output_training_file_template))
                batch_examples_train = []

        batch_examples_val = []
        for step in xrange(len(validation_filenames)*10):
            print step
            val_data = sess.run(validation_data_batch)
            _, im = decode_data_stream(val_data, dictionary, 100, metadata)
            vgg_features = sess.run(vgg.fc2, feed_dict={vgg.imgs: im})
            batch_examples_val += write_example_batch(val_data, vgg_features, validation_batch_size)
            if step % 10 == 9:
                write_batch(batch_examples_val, os.path.join(output_dir, output_validation_file_template))
                batch_examples_val = []




def write_example_batch(text_data,vgg_features,batchsize):
    batch_examples = []
    for i in xrange(batchsize):
        feature_data = dict()
        feature_data['label'] = _bytes_feature(text_data[3][1][i])
        image_field = '{}/{}/{}'.format('image', 'VGG_features', 'encoded')
        feature_data[image_field] = _floats_feature(vgg_features[i].astype(float))
        name_raw = '{}/{}/{}'.format('text', 'title', 'raw')
        feature_data[name_raw] = _bytes_feature(str(text_data[0][1][i].encode('ascii', 'ignore').decode('ascii')) if text_data[0][1][i] else '')
        name_raw = '{}/{}/{}'.format('text', 'description', 'raw')
        feature_data[name_raw] = _bytes_feature(str(text_data[1][1][i].encode('ascii', 'ignore').decode('ascii')) if text_data[1][1][i] else '')
        batch_examples.append(tf.train.Example(features=tf.train.Features(feature=feature_data)))
    return batch_examples

def write_batch(examples, output_file_template):
    count = 1
    while os.path.isfile(output_file_template.format(count)):
        count += 1

    writer = tf.python_io.TFRecordWriter(output_file_template.format(count))
    for example in examples:
        writer.write(example.SerializeToString())
    writer.close()


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def train_vgg(max_steps, train_dir, restore_from_ckpt, checkpoint_dir, train_filenames=[], validation_filenames=[],
          network_spec=simple_conv_word_embedding, log_device_placement=True, size_embedding=100, vocab_size=None,
          min_queue_examples=10,
          metadata=[], dictionary=None):

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement))
    train_batch_size = metadata.train_batch_size
    validation_batch_size = metadata.validation_batch_size
    imgs = tf.placeholder(tf.float32, [metadata.train_batch_size, metadata.imsize, metadata.imsize, 3])  #
    vgg = vgg16(imgs)
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

    with tf.variable_scope("model", reuse=None, initializer=initializer):

        single_data_source, label = network_spec.read_and_decode_sentence(train_filename_queue, metadata)
        train_data_batch = generate_sentence_and_label_batch(single_data_source, label,
                                                             min_queue_examples=min_queue_examples,
                                                             batch_size=train_batch_size)
        text = tf.placeholder(tf.int32, shape=(train_batch_size, 40))
        extras = tf.placeholder(tf.int32, shape=(train_batch_size, 40, 3))
        labels_data_train = tf.placeholder(tf.int32, shape=(train_batch_size, metadata.num_classes))
        logits, last_hidden = vgg.inference_vgg()
        title_logits, _, _ = vgg.title_inference(text, extras, vocab_size, size_embedding, metadata=metadata,training=False)
        if metadata.trainpolicy:
            #im_conf = tf.expand_dims(tf.reduce_max(tf.nn.softmax(logits),1),[1])
            #title_conf = tf.expand_dims(tf.reduce_max(tf.nn.softmax(title_logits),1),[1])
            title_conf = tf.nn.top_k(tf.nn.softmax(title_logits), k=3)[0]
            im_conf = tf.nn.top_k(tf.nn.softmax(logits), k=3)[0]
            # im_conf = tf.nn.softmax(logits)
            # title_conf = tf.nn.softmax(title_logits)
            policy_logits = vgg.policy_inference(text, extras, vocab_size, size_embedding,im_conf,title_conf,3, metadata)
            policy =  tf.squeeze(tf.round(tf.nn.sigmoid(policy_logits)))
            # policy = tf.cast(tf.greater(im_conf,title_conf ),'float')
            accuracy_im, acc_vec_im = get_accuracy(logits, labels_data_train)
            accuracy_title, acc_vec_title = get_accuracy(title_logits, labels_data_train)
            # policy_target = tf.nn.relu(tf.cast(acc_vec_title-acc_vec_im, tf.float32))
            policy_target = tf.nn.relu(tf.cast(acc_vec_im-acc_vec_title, tf.float32))
            accuracy_vec_policy = tf.add(tf.mul(policy,acc_vec_im),tf.mul(tf.square(tf.sub(1.0,policy)),acc_vec_title))
            # accuracy_vec_policy = tf.add(tf.mul(policy,acc_vec_title),tf.mul(tf.square(tf.sub(1.0,policy)),acc_vec_im))
            accuracy_policy = tf.reduce_mean(accuracy_vec_policy)
            accuracy_opt = tf.reduce_mean(tf.cast(tf.logical_or(tf.cast(acc_vec_title,tf.bool),tf.cast(acc_vec_im,tf.bool)),tf.float32))
            policy_pred = tf.reduce_mean(tf.cast(tf.equal(policy,policy_target),tf.float32))

        else:
            accuracy,_ = get_accuracy(logits+title_logits, labels_data_train)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        # Get images and labels for CIFAR-10.
        single_data_source_val, label_val = network_spec.read_and_decode_sentence(validation_filename_queue, metadata)
        validation_data_batch = generate_sentence_and_label_batch(single_data_source_val, label_val,
                                                                  min_queue_examples=min_queue_examples,
                                                                  batch_size=validation_batch_size)
    #     text_val = tf.placeholder(tf.int32, shape=(train_batch_size, 40))
    #     extras_val = tf.placeholder(tf.int32, shape=(train_batch_size, 40, 3))
    #     labels_data_val = tf.placeholder(tf.int32, shape=(validation_batch_size, metadata.num_classes))
    #     logits_val, _ = vgg.inference_vgg()
    #     title_logits_val,_,_ = vgg.title_inference(text_val, extras_val, vocab_size, size_embedding, metadata=metadata,training=False)
    #     if metadata.trainpolicy:
    #         im_conf_val = tf.expand_dims(tf.reduce_max(tf.nn.softmax(logits_val), 1), [1])
    #         title_conf_val = tf.expand_dims(tf.reduce_max(tf.nn.softmax(title_logits_val), 1), [1])
    #         policy_logits_val = vgg.policy_inference(text_val, extras_val, vocab_size, size_embedding,im_conf_val,title_conf_val,metadata=metadata,training=False)
    #         policy_val = tf.squeeze(tf.round(tf.nn.sigmoid(policy_logits_val)))
    #         accuracy_val_im, acc_val_vec_im = get_accuracy(logits_val, labels_data_val)
    #         accuracy_val_title, acc_val_vec_title = get_accuracy(title_logits_val, labels_data_val)
    #         policy_target_val = tf.nn.relu(tf.cast(acc_val_vec_im-acc_val_vec_title ,tf.float32))
    #         accuracy_vec_policy_val = tf.add(tf.mul(policy_val,acc_val_vec_im),tf.mul(tf.square(tf.sub(1.0,policy_val)),acc_val_vec_title))
    #         accuracy_val_policy = tf.reduce_mean(accuracy_vec_policy_val)
    #         accuracy_opt_val = tf.reduce_mean(tf.cast(tf.logical_or(tf.cast(acc_val_vec_im, tf.bool), tf.cast(acc_val_vec_title, tf.bool)), tf.float32))
    #         policy_pred_val = tf.reduce_mean(tf.cast(tf.equal(policy_val, policy_target_val), tf.float32))
    #
    #     else:
    #         accuracy_val,acc_val_vec = get_accuracy(logits_val+title_logits_val, labels_data_val)
    # # Calculate loss.


    if metadata.trainpolicy:
        #used for averaging
        accuracy_value = tf.placeholder(tf.float32, shape=5)
        accuracy_value_val = tf.placeholder(tf.float32, shape=5)

        # title accuracy
        accuracy_log, accuracy_avg_log = log_accuracy(accuracy_value[1], summary_postfix='title_train')
        accuracy_val_log, accuracy_avg_val_log = log_accuracy(accuracy_value_val[1], summary_postfix='title_validation')

        # image accuracy
        accuracy_log, accuracy_avg_log = log_accuracy(accuracy_value[0], summary_postfix='image_train')
        accuracy_val_log, accuracy_avg_val_log = log_accuracy(accuracy_value_val[0], summary_postfix='image_validation')

        # policy accuracy
        accuracy_log, accuracy_avg_log = log_accuracy(accuracy_value[2], summary_postfix='policy_train')
        accuracy_val_log, accuracy_avg_val_log = log_accuracy(accuracy_value_val[2], summary_postfix='policy_validation')

        accuracy_log, accuracy_avg_log = log_accuracy(accuracy_value[3], summary_postfix='opt_train')
        accuracy_val_log, accuracy_avg_val_log = log_accuracy(accuracy_value_val[3],summary_postfix='opt_validation')

        accuracy_log, accuracy_avg_log = log_accuracy(accuracy_value[4], summary_postfix='policy_pred_train')
        accuracy_val_log, accuracy_avg_val_log = log_accuracy(accuracy_value_val[4], summary_postfix='policy_pred_validation')

        loss = network_spec.policy_loss(tf.squeeze(policy_logits), policy_target,metadata)
    else:
        accuracy_value = tf.placeholder(tf.float32, shape=1)
        accuracy_value_val = tf.placeholder(tf.float32, shape=1)
        accuracy_log, accuracy_avg_log = log_accuracy(accuracy_value, summary_postfix='train')
        accuracy_val_log, accuracy_avg_val_log = log_accuracy(accuracy_value_val, summary_postfix='validation')
        loss = network_spec.loss(logits+title_logits, labels_data_train, metadata)
    global_step = tf.Variable(0, trainable=False)
    train_op = network_spec.train(loss, global_step, metadata=metadata)
    saver = tf.train.Saver(tf.all_variables())
    summary_op = tf.merge_all_summaries()

    # Initialize model/embedding first before initializing anything else.
    print [v.name for v in tf.all_variables()]
    vars = [v for v in tf.all_variables()]
    print 'Variables: ', len(vars)
    init = tf.initialize_variables(vars)
    sess.run(init)

    if metadata.image_flag or metadata.logits_flag:
        if not hasattr(metadata, 'vgg_checkpoint'):
            vgg.load_weights(metadata.vgg_weights, sess)
    load_mix_network(metadata, restore_from_ckpt, checkpoint_dir, sess)
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(train_dir, graph_def=sess.graph_def)
    for step in xrange(max_steps):
        start_time = time.time()
        data_train = sess.run(train_data_batch)
        train_data_concat, im = decode_data_stream(data_train, dictionary, train_batch_size, metadata)
        _, loss_value = sess.run([train_op, loss],feed_dict={imgs:im,labels_data_train:train_data_concat[2],text:train_data_concat[0],extras:train_data_concat[1]})
        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if (step + 1) % 10 == 1:
            num_examples_per_step = train_batch_size
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)
            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f ''sec/batch)')
            print (format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

        if (step + 1) % 100 == 1:
            if metadata.trainpolicy:
                acc_train_list = np.zeros(shape=(metadata.validation_iters,5))
                acc_val_list = np.zeros(shape=(metadata.validation_iters,5))

                # compute multiple batches of acc to get more robust results
                for i in xrange(metadata.validation_iters):
                    data_train = sess.run(train_data_batch)
                    train_data_concat, im = decode_data_stream(data_train, dictionary, train_batch_size, metadata)
                    acc = sess.run([accuracy_im,accuracy_title,accuracy_policy,accuracy_opt,policy_pred],
                                   feed_dict={imgs: im, labels_data_train: train_data_concat[2],
                                              text: train_data_concat[0], extras: train_data_concat[1]})

                    data_val = sess.run(validation_data_batch)
                    val_data_concat, im_val = decode_data_stream(data_val, dictionary, validation_batch_size, metadata)
                    # acc_val = sess.run([accuracy_val_im,accuracy_val_title,accuracy_val_policy,accuracy_opt_val,policy_pred_val]
                    #                    ,feed_dict={imgs: im_val, labels_data_val: val_data_concat[2],
                    #                                 text_val: val_data_concat[0], extras_val: val_data_concat[1]})
                    acc_val = sess.run(
                        [accuracy_im, accuracy_title, accuracy_policy, accuracy_opt, policy_pred]
                        , feed_dict={imgs: im_val, labels_data_train: val_data_concat[2],
                                     text: val_data_concat[0], extras: val_data_concat[1]})
                    acc_train_list[i]=acc
                    acc_val_list[i]=acc_val


                acc_avg_train = acc_train_list.mean(axis=0)
                acc_avg_val = acc_val_list.mean(axis=0)

                print '[image_acc,title_acc,policy_acc,opt_acc,policy_pred_acc]'
                print acc_avg_train
                print 'val:[image_acc,title_acc,policy_acc,opt_acc,policy_pred_acc]:'
                print acc_avg_val

                # summary_str = sess.run(
                #     [summary_op, accuracy_log, accuracy_avg_log, accuracy_val_log, accuracy_avg_val_log],
                #     feed_dict={accuracy_value: acc_avg_train, labels_data_val: val_data_concat[2],
                #                imgs: im, labels_data_train: train_data_concat[2], accuracy_value_val: acc_avg_val,
                #                text: train_data_concat[0], extras: train_data_concat[1], text_val: val_data_concat[0],
                #                extras_val: val_data_concat[1]})
                summary_str = sess.run(
                    [summary_op, accuracy_log, accuracy_avg_log, accuracy_val_log, accuracy_avg_val_log],
                    feed_dict={accuracy_value: acc_avg_train,imgs: im, labels_data_train: train_data_concat[2],
                               accuracy_value_val: acc_avg_val,text: train_data_concat[0], extras: train_data_concat[1],
                               })
            else:
                # Evaluate the model, compute accuracy
                acc_train_list = []
                acc_val_list = []
                # compute multiple batches of acc to get more robust results
                for i in xrange(metadata.validation_iters):
                    data_train = sess.run(train_data_batch)
                    train_data_concat, im = decode_data_stream(data_train, dictionary, train_batch_size, metadata)
                    acc = sess.run([accuracy],feed_dict={imgs:im,labels_data_train:train_data_concat[2],text:train_data_concat[0],extras:train_data_concat[1]})
                    data_val = sess.run(validation_data_batch)
                    val_data_concat, im_val = decode_data_stream(data_val, dictionary, validation_batch_size, metadata)
                    # acc_val = sess.run([accuracy_val], feed_dict={imgs: im_val, labels_data_val: val_data_concat[2],text_val:val_data_concat[0],extras_val:val_data_concat[1]})
                    acc_train_list.append(acc[0])
                    acc_val_list.append(acc_val[0])

                acc_avg_train = np.zeros(shape=1)
                acc_avg_train[0] = sum(acc_train_list) / len(acc_train_list)
                acc_avg_val = np.zeros(shape=1)
                acc_avg_val[0] = sum(acc_val_list) / len(acc_val_list)

                print acc_avg_train
                print acc_train_list
                print acc_avg_val
                print acc_val_list

                # summary_str = sess.run([summary_op, accuracy_log, accuracy_avg_log,accuracy_val_log, accuracy_avg_val_log],
                #                        feed_dict={accuracy_value: acc_avg_train,labels_data_val: val_data_concat[2],
                #                                     imgs:im, labels_data_train:train_data_concat[2],accuracy_value_val: acc_avg_val,
                #                                   text: train_data_concat[0], extras: train_data_concat[1],text_val:val_data_concat[0],extras_val:val_data_concat[1]})
            summary_writer.add_summary(summary_str[0], step)


        if (step + 1) % 1000 == 10 or (step + 1) == max_steps:
            checkpoint_path = os.path.join(train_dir, 'model_{}.ckpt'.format(step))
            saver.save(sess, checkpoint_path, global_step=step)




def compare_models_tsne(file1,file2):
    import pickle

    with open('/Users/tzahavy/git/code/tensorflow-classifiers/tsne_dat_' + file1 + '.pkl', 'rb') as pickle_load:
        Y = pickle.load(pickle_load)
    with open('/Users/tzahavy/git/code/tensorflow-classifiers/tsne_acc_' + file1 + '.pkl', 'rb') as pickle_load:
        acc = pickle.load(pickle_load)
    with open('/Users/tzahavy/git/code/tensorflow-classifiers/tsne_labels_' + file1 + '.pkl', 'rb') as pickle_load:
        pred = pickle.load(pickle_load)

    with open('/Users/tzahavy/git/code/tensorflow-classifiers/tsne_acc_' + file2 + '.pkl', 'rb') as pickle_load:
        acc2 = pickle.load(pickle_load)

    tsne_plot(Y, acc, pred,acc2)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out



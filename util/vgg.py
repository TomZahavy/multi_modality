########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import tensorflow as tf
import numpy as np
# from scipy.misc import imread, imresize
# from imagenet_classes import class_names
from conv_net.simple_conv_word_embedding import _variable_with_weight_decay,_activation_summary

class vgg16:
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.convlayers()
        self.fc_layers()
        self.inference_vgg()
        self.features = self.fc2
        self.logits = self.softmax
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)


    def convlayers(self):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        # fc3
        # with tf.name_scope('fc3') as scope:
        #     fc3w = tf.Variable(tf.truncated_normal([4096, 1000],
        #                                                  dtype=tf.float32,
        #                                                  stddev=1e-1), name='weights')
        #     fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
        #                          trainable=True, name='biases')
        #     self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
        #     self.parameters += [fc3w, fc3b]

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        count = 0
        for i, k in enumerate(keys):
            if i==30: ## dont need the last layer
                break
            print i, k, np.shape(weights[k])
            sess.run(self.parameters[i].assign(weights[k]))
            count += np.size(weights[k])

    def inference_vgg(self, training=True):

        if training:
            with tf.name_scope("dropout"):
                h_dropout = tf.nn.dropout(self.fc2, 1) # change one to metadata.dropout
                last_hidden = h_dropout
        else:
            last_hidden = self.fc2
        _activation_summary(last_hidden)

        with tf.variable_scope('softmax_linear') as scope:
            weights = _variable_with_weight_decay(name='softmax_linear_weights', shape=[self.fc2._shape[1].value,
                                                  2890], init_type='truncated_normal',
                                                  init_val=1e-1, wd=[])
            biases = _variable_with_weight_decay(name='softmax_linear_biases', shape=[2890],
                                                 init_type='const', init_val=1.0, wd=[])
            softmax_linear = tf.nn.xw_plus_b(last_hidden, weights, biases, name=scope.name)

            _activation_summary(softmax_linear)
        self.softmax = softmax_linear
        return softmax_linear, last_hidden

    def title_inference(self,input_ids, extras, vocab_size, size_embedding,
                  extra_size=3, metadata=[], training=True):

        # Text cnn
        last_hidden,num_filters_total,h_pool = self.vgg_input_to_last_hidden(input_ids, extras, vocab_size, size_embedding, extra_size,
                                                                      metadata, training)
        penalty = metadata.penalty
        with tf.variable_scope('title_softmax_linear') as scope:
            weights = _variable_with_weight_decay(name='softmax_linear_weights', shape=[num_filters_total,
                                                                                        metadata.num_classes],
                                                  init_type='xavier', wd=penalty)
            biases = _variable_with_weight_decay(name='softmax_linear_biases', shape=[metadata.num_classes],
                                                 init_type='const', init_val=0.1, wd=penalty)
            softmax_linear = tf.nn.xw_plus_b(last_hidden, weights, biases, name=scope.name)

            _activation_summary(softmax_linear)

        return softmax_linear, last_hidden, h_pool

    def policy_inference(self, input_ids, extras, vocab_size, size_embedding,im_conf,title_conf,extra_size=3, metadata=[], training=True):

        with tf.variable_scope('policy'):
            # last_hidden, num_filters_total, h_pool = self.vgg_input_to_last_hidden(input_ids, extras, vocab_size, size_embedding,
            #                                                                extra_size, metadata, training,metadata.dropout,metadata.num_filters)
            # last_hidden = tf.concat(1, [last_hidden, im_conf])
            # last_hidden = tf.concat(1, [last_hidden, title_conf])
            last_hidden = im_conf
            last_hidden = tf.concat(1, [last_hidden, title_conf])

            penalty = metadata.penalty2
            h_size = 10
            with tf.variable_scope('policy_softmax_linear') as scope:
                weights = _variable_with_weight_decay(name='policy_softmax_linear_weights', shape=[last_hidden._shape[1]._value,h_size],
                                              init_type='xavier', wd=penalty)
                biases = _variable_with_weight_decay(name='policy_softmax_linear_biases', shape=[h_size],
                                             init_type='const', init_val=metadata.policy_logits_bias, wd=penalty)

                h = tf.nn.relu(tf.nn.xw_plus_b(last_hidden, weights, biases, name=scope.name))

                weights_2 = _variable_with_weight_decay(name='policy_softmax_linear_weights_2',
                                                      shape=[h_size,1],
                                                      init_type='xavier', wd=penalty)
                biases_2 = _variable_with_weight_decay(name='policy_softmax_linear_biases_2', shape=[1],
                                                 init_type='const', init_val=metadata.policy_logits_bias, wd=penalty)
                softmax_linear = tf.nn.xw_plus_b(h, weights_2, biases_2, name=scope.name)

            _activation_summary(softmax_linear)

        return softmax_linear

    def vgg_input_to_last_hidden(self,input_ids, extras, vocab_size, size_embedding,
                                 extra_size, metadata, training,dropprob=1,numfilters=128):

        # Text cnn
        embedding = tf.get_variable("embedding", [vocab_size, size_embedding], dtype=tf.float32)
        num_filters_total = 0
        pooled_outputs = []
        text_counter = 0

        input = input_ids
        extra = extras
        data_source = metadata.data_sources[0] # a bit messy, change later
        filter_sizes = data_source.filters
        num_filters_total += numfilters * len(filter_sizes)
        pooled_outputs += self.vgg_conv_layer(embedding, size_embedding, input, extra, filter_sizes, extra_size, metadata,
                                             data_source,numfilters)

        h_pool = tf.concat(1, pooled_outputs, name='h_pool')
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        if training:
            with tf.name_scope("dropout"):
                h_dropout = tf.nn.dropout(h_pool_flat, dropprob)

        # Add linear fully conected layers
        if training:
            with tf.name_scope("dropout"):
                inputs = [h_dropout]
        else:
            inputs = [h_pool_flat]
        _activation_summary(inputs[0])


        return inputs[0], num_filters_total, inputs[0]
    def vgg_conv_layer(self,embedding,size_embedding,input_ids,extras,filter_sizes,extra_size,metadata,data_source,numfilters):
        #Text cnn
        inputs = tf.expand_dims(tf.concat(2, [tf.nn.embedding_lookup(embedding, input_ids), tf.cast(extras,dtype=tf.float32)]), -1)
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            scope_name = "conv-maxpool-{}-{}".format(filter_size,data_source.name)
            print 'working with filter size: {}, scope_name: {}'.format(filter_size, scope_name)
            with tf.variable_scope(scope_name) as scope:
                # Convolution Layer
                filter_shape = [filter_size, size_embedding + extra_size, 1, numfilters]
                W = _variable_with_weight_decay(name='W', shape=filter_shape,
                                                      init_type='xavier', wd=metadata.penalty)
                print 'name of W: {}'.format(W.name)


                conv = tf.nn.conv2d(
                    inputs,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(conv, name="relu")
                h_bn = tf.contrib.layers.batch_norm(h)
                # Max-pooling over the outputs
                pooled = tf.nn.max_pool(
                    h_bn,
                    ksize=[1, data_source.size - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                # pooled_outputs.append(tf.nn.softmax(pooled[:,0,0,:]))
                pooled_outputs.append(pooled[:,0,0,:])

        return pooled_outputs
if __name__ == '__main__':
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, 'vgg16_weights.npz', sess)

    # img1 = imread('laska.png')
    # img1 = imresize(img1, (224, 224))

    # prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
    # preds = (np.argsort(prob)[::-1])[0:5]
    # for p in preds:
    #     print class_names[p], prob[p]
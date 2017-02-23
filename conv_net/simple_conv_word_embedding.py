import re
import tensorflow as tf
TOWER_NAME = 'tower'
MOVING_AVERAGE_DECAY = 0.9999
EMBEDDING_PADDED = 'embedding_padded'
EXTRA_EMBEDDING_FEATURES = 'embedding_extra'
import numpy as np
def _variable_with_weight_decay(name, shape, init_type, init_val=0, wd=[]):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    print 'name: {}, shape: {}'.format(name, shape)
    if init_type == 'xavier':
        var = tf.get_variable(name, shape,
                              initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
    elif init_type == 'truncated_normal':
        var = tf.get_variable(name, shape,
                              initializer=tf.truncated_normal_initializer(stddev=init_val), dtype=tf.float32)
    elif init_type == 'const':
        var = tf.get_variable(name, shape,
                              initializer=tf.constant_initializer(init_val), dtype=tf.float32)
    if len(wd)>0:
        weight_decay_l2 = tf.mul(tf.nn.l2_loss(var), wd[1], name='weight_loss_l2')
        weight_decay_l1 = tf.mul(tf.reduce_sum(tf.abs(var)),wd[0] , name='weight_loss_l1')
        tf.add_to_collection('losses', weight_decay_l1)
        tf.add_to_collection('losses', weight_decay_l2)

    return var


def read_and_decode_sentence(filename_queue, metadata):

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features_dic = {'label': tf.VarLenFeature(dtype=tf.string)}

    for data_source in metadata.data_sources:
        if data_source.name == 'image' or data_source.name == 'image_logits':
            image_field = '{}/{}/{}'.format('image', 'Primary Image', 'encoded')
            features_dic[image_field] = tf.FixedLenFeature([], dtype=tf.string, default_value='')
        elif data_source.name == 'image_features':
            image_field = '{}/{}/{}'.format('image', 'VGG_features', 'encoded')
            features_dic[image_field] = tf.VarLenFeature(dtype=tf.float32)#tf.FixedLenFeature([], dtype=tf.float32, default_value=0)
        else:
            name_raw = '{}/{}/{}'.format('text', data_source.name, 'raw')
            features_dic[name_raw] = tf.VarLenFeature(dtype=tf.string)

    features = tf.parse_single_example(serialized_example, features=features_dic)
    data = []
    for data_source in metadata.data_sources:
        if data_source.name == 'image' or data_source.name == 'image_logits':
            data.append(tf.cast(tf.image.resize_images(tf.image.decode_jpeg(features[image_field],channels=3),metadata.imsize,metadata.imsize),tf.float32))
            # data.append(tf.image.resize_images(tf.image.decode_jpeg(features[image_field],channels=3),500,500))

        elif data_source.name == 'image_features':
            data.append(features[image_field])
        else:
            name_raw = '{}/{}/{}'.format('text', data_source.name, 'raw')
            data.append(features[name_raw])
    label = tf.cast(features['label'], tf.string)

    return data,label


def generate_sentence_and_label_batch(data, label, min_queue_examples, batch_size=100):
    num_preprocess_threads = 16

    inputs = data + [label]
    data = tf.train.shuffle_batch(
        inputs,
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)

    return data

def generate_sentence_and_label_batch_noshuffle(data, label, min_queue_examples, batch_size=100):
    num_preprocess_threads = 1

    inputs = data + [label]
    data = tf.train.batch(
        inputs,
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

    return data

def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def inference(input_ids, extras, vocab_size, size_embedding,
              extra_size=3,image_features=[], metadata=[], training=True):

    last_hidden, num_filters_total,h_pool = input_to_last_hidden(input_ids, extras,vocab_size,size_embedding,extra_size,
                                                                 metadata, training,image_features)
    penalty = metadata.penalty
    with tf.variable_scope('softmax_linear') as scope:

        weights = _variable_with_weight_decay(name='softmax_linear_weights', shape=[num_filters_total,
                                              metadata.num_classes], init_type='xavier', wd=penalty)
        biases = _variable_with_weight_decay(name='softmax_linear_biases', shape=[metadata.num_classes],
                                         init_type='const', init_val=0.1, wd=penalty)
        softmax_linear = tf.nn.xw_plus_b(last_hidden, weights, biases, name=scope.name)

        _activation_summary(softmax_linear)

    return softmax_linear,last_hidden,h_pool


def input_to_last_hidden(input_ids, extras, vocab_size, size_embedding,
                         extra_size, metadata,training,image_features):
    if metadata.text_flag:
        embedding = tf.get_variable("embedding", [vocab_size, size_embedding], dtype=tf.float32)

    num_filters_total = 0
    pooled_outputs = []
    text_counter = 0
    for data_source in metadata.data_sources:
        if data_source.name == 'image' or data_source.name == 'image_features':
            pooled_outputs.append(image_features[0])
            num_filters_total += data_source.size
        elif data_source.name == 'image_logits' :
            continue
        else:
            input = input_ids[text_counter]
            extra= extras[text_counter]
            text_counter+=1
            filter_sizes = data_source.filters
            num_filters_total += metadata.num_filters * len(filter_sizes)
            pooled_outputs += conv_layer(embedding,size_embedding,input,extra,filter_sizes,extra_size,metadata,data_source)

    # Combine all the pooled features with/out a gate function
    if metadata.gate == 'gate1':
        num_filters_total = metadata.gate_size
        text_concat = tf.concat(1, pooled_outputs[0:-1], name='h_pool')
        h_pool = inputgate(text_concat,pooled_outputs[-1],metadata.gate_size)
    elif metadata.gate == 'gate2':
        num_filters_total = metadata.gate_size
        text_concat = tf.concat(1, pooled_outputs[0:-1], name='h_pool')
        h_pool = inputgate2(text_concat, pooled_outputs[-1], metadata.gate_size)
    elif metadata.gate == 'gate3':
        num_filters_total = metadata.gate_size
        text_concat = tf.concat(1, pooled_outputs[0:-1], name='h_pool')
        h_pool = inputgate3(text_concat, pooled_outputs[-1], metadata.gate_size)
    else:
        h_pool = tf.concat(1, pooled_outputs,name='h_pool')
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    if training:
        with tf.name_scope("dropout"):
            h_dropout = tf.nn.dropout(h_pool_flat, metadata.dropout)


    # Add linear fully conected layers
    Nlayers = metadata.Nlayers
    if training:
        with tf.name_scope("dropout"):
            inputs = [h_dropout]
    else:
        inputs = [h_pool_flat]
    _activation_summary(inputs[0])

    for i in xrange(Nlayers):
        scope_name = "fully_connected-{}".format(i)
        with tf.variable_scope(scope_name) as scope:
            weights = _variable_with_weight_decay(name='weights', shape=[num_filters_total, num_filters_total],
                                                  init_type='xavier')
            biases = tf.get_variable('biases', [num_filters_total],
                                 initializer=tf.constant_initializer(0.0))
            ### ReLu + linear
            h = tf.nn.relu(tf.nn.xw_plus_b(inputs[i], weights, biases, name=scope.name), name="relu")

            ### Dropout
            if training:
                h_dropout = tf.nn.dropout(h, metadata.dropout,name='dropout')
                inputs.append(h_dropout)
            else:
                inputs.append(h)

            _activation_summary(inputs[i+1])
    return inputs[Nlayers], num_filters_total,inputs[0]


def conv_layer(embedding,size_embedding,input_ids,extras,filter_sizes,extra_size,metadata,data_source):

    inputs = tf.expand_dims(tf.concat(2, [tf.nn.embedding_lookup(embedding, input_ids), tf.cast(extras,dtype=tf.float32)]), -1)
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        scope_name = "conv-maxpool-{}-{}".format(filter_size,data_source.name)
        print 'working with filter size: {}, scope_name: {}'.format(filter_size, scope_name)
        with tf.variable_scope(scope_name) as scope:
            # Convolution Layer
            filter_shape = [filter_size, size_embedding + extra_size, 1, metadata.num_filters]
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



def get_accuracy(logits, labels):
    correct_predictions = tf.equal(tf.argmax(logits,1),tf.argmax(tf.mul(tf.cast(labels,'float32'), logits), 1))
    accuracy = tf.cast(correct_predictions,tf.float32)
    return tf.reduce_mean(accuracy),accuracy

def log_accuracy(acc_list,summary_postfix):
    acc = tf.reduce_mean(acc_list)
    accuracy_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    avg_accuracy_op = accuracy_averages.apply([acc])
    with tf.control_dependencies([acc, avg_accuracy_op]):
        tf.scalar_summary("accuracy_avg_{}".format(summary_postfix), accuracy_averages.average(acc))
        tf.scalar_summary("accuracy_{}".format(summary_postfix), acc)
        accuracy_op = tf.no_op(name='accuracy_op')

    return accuracy_op,avg_accuracy_op

def loss(logits, labels, metadata):
    cross_entropy = tf.nn.weighted_cross_entropy_with_logits(
        tf.squeeze(logits), tf.cast(labels, tf.float32),metadata.pos_coeff,name='cross_entropy_per_example')

    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def policy_loss(policy_logits, policy_target,metadata):
    cross_entropy = tf.nn.weighted_cross_entropy_with_logits(policy_logits, policy_target,metadata.policy_pos_coeff)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(l.op.name + ' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step,metadata=[]):

    loss_averages_op = _add_loss_summaries(total_loss)
    with tf.control_dependencies([loss_averages_op]):
        # opt = tf.train.GradientDescentOptimizer(lr)
        if metadata.optimizer == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate=metadata.lr, beta1=0.9, beta2=0.999, epsilon=1e-08) #0.001
        elif metadata.optimizer == 'mom':
            opt = tf.train.MomentumOptimizer(metadata.lr, 0.9)
        else:
            opt = tf.train.GradientDescentOptimizer(metadata.lr)
        if metadata.trainpolicy:
            vars = [v for v in tf.all_variables() if ('policy' in v.name)]
            grads_and_vars = opt.compute_gradients(total_loss,vars)
        else:
            grads_and_vars = opt.compute_gradients(total_loss)


    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads_and_vars, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads_and_vars:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import tensor_shape

def dropout_inputpipes(x, indices,keep_prob,name=None,seed=None):
  """Computes dropout.

  With probability `keep_prob`, outputs the input element scaled up by
  `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
  sum is unchanged.

  By default, each element is kept or dropped independently.  If `noise_shape`
  is specified, it must be
  [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
  will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
  and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
  kept independently and each row and column will be kept or not kept together.

  Args:
    x: A tensor.
    keep_prob: A scalar `Tensor` with the same type as x. The probability
      that each element is kept.
    noise_shape: A 1-D `Tensor` of type `int32`, representing the
      shape for randomly generated keep/drop flags.
    seed: A Python integer. Used to create random seeds. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    name: A name for this operation (optional).

  Returns:
    A Tensor of the same shape of `x`.

  Raises:
    ValueError: If `keep_prob` is not in `(0, 1]`.
  """
  with ops.op_scope([x], name, "dropout") as name:
    input_keep_prob = 1/float(len(indices))
    x = ops.convert_to_tensor(x, name="x")
    if isinstance(keep_prob, float) and not 0 < keep_prob <= 1:
      raise ValueError("keep_prob must be a scalar tensor or a float in the "
                       "range (0, 1], got %g" % keep_prob)
    keep_prob = ops.convert_to_tensor(keep_prob,
                                      dtype=x.dtype,
                                      name="keep_prob")
    keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

    binary_tensors = []
    for ind in indices:
        # noise_shape = array_ops.shape(x)
        # noise_shape[1] = array_ops.shape(ind)
        # uniform [keep_prob, 1.0 + keep_prob)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform((array_ops.shape(x)[0],ind),
                                                   seed=seed,
                                                   dtype=x.dtype)
        input_random_tensor = input_keep_prob
        input_random_tensor += random_ops.random_uniform((array_ops.shape(input_keep_prob)),
                                                   seed=seed,
                                                   dtype=x.dtype)
        # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        binary_tensors.append(math_ops.floor(random_tensor)*math_ops.floor(input_random_tensor))
    binary_tensor = tf.concat(1,binary_tensors)
    ret = x * math_ops.inv(keep_prob*input_keep_prob) * binary_tensor
    ret.set_shape(x.get_shape())
    return ret


def inputgate(x,y, size, carry_bias=-1.0):
    # gets two inputs, choosing a bernouli mask between them
  W_Tx = tf.Variable(tf.truncated_normal([size, size], stddev=0.1), name="weight_transform")
  W_Ty = tf.Variable(tf.truncated_normal([size, size], stddev=0.1), name="weight_transform2")
  b_T = tf.Variable(tf.constant(carry_bias, shape=[size]), name="bias_transform")
  T = tf.sigmoid(tf.matmul(x, W_Tx) + tf.matmul(y, W_Ty)+ b_T, name="transform_gate")
  C = tf.sub(1.0, T, name="carry_gate")

  z = tf.add(tf.mul(y, T), tf.mul(x, C), "z")

  return z


def inputgate2(x, y,size, carry_bias=-1.0):
    x_size = x._shape[1]._value
    y_size = y._shape[1]._value
    # gets two inputs, choosing a bernouli mask between them
    W_Tx = tf.Variable(tf.truncated_normal([x_size, 1], stddev=0.1), name="weight_transform")
    W_Ty = tf.Variable(tf.truncated_normal([y_size, 1], stddev=0.1), name="weight_transform2")
    b_T = tf.Variable(tf.constant(carry_bias, shape=[1]), name="bias_transform")

    T = tf.sigmoid(tf.matmul(x, W_Tx) + tf.matmul(y, W_Ty) + b_T, name="transform_gate")
    C = tf.sub(1.0, T, name="carry_gate")

    W_x = tf.Variable(tf.truncated_normal([x_size, size], stddev=0.1), name="weight_transform")
    W_y = tf.Variable(tf.truncated_normal([y_size, size], stddev=0.1), name="weight_transform2")
    b = tf.Variable(tf.constant(carry_bias, shape=[size]), name="bias_transform")

    z = tf.matmul(y, W_y)*T+tf.matmul(x, W_x)*C +b

    return z

def inputgate3(text, image,size, carry_bias=-1.0):
    x_size = text._shape[1]._value
    y_size = image._shape[1]._value
    # gets two inputs, choosing a bernouli mask between them
    W_Ty = tf.Variable(tf.truncated_normal([y_size, 1], stddev=0.1), name="weight_transform2")
    b_T = tf.Variable(tf.constant(carry_bias, shape=[1]), name="bias_transform")

    T = tf.sigmoid(tf.matmul(image, W_Ty) + b_T, name="transform_gate")
    C = tf.sub(1.0, T, name="carry_gate")

    W_x = tf.Variable(tf.truncated_normal([x_size, size], stddev=0.1), name="weight_transform")
    W_y = tf.Variable(tf.truncated_normal([y_size, size], stddev=0.1), name="weight_transform2")
    b = tf.Variable(tf.constant(carry_bias, shape=[size]), name="bias_transform")

    z = tf.matmul(image, W_y)*T+tf.matmul(text, W_x)*C +b

    return z

def inference2(input_ids, extras, vocab_size, size_embedding,
                  extra_size=3, image_features=[], metadata=[], training=True):


    last_hidden, num_filters_total, h_pool = input_to_last_hidden(input_ids, extras, vocab_size, size_embedding, extra_size,
                                                                  metadata, training, [])
    penalty = metadata.penalty
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay(name='softmax_linear_weights', shape=[num_filters_total,
                                                                                    metadata.num_classes],
                                              init_type='xavier', wd=penalty)
        biases = _variable_with_weight_decay(name='softmax_linear_biases', shape=[metadata.num_classes],
                                             init_type='const', init_val=0.1, wd=penalty)
        # softmax_linear = inputgate(tf.nn.xw_plus_b(last_hidden, weights, biases, name=scope.name),image_features[0],metadata.num_classes)
        softmax_linear = tf.add(tf.nn.xw_plus_b(last_hidden, weights, biases, name=scope.name),image_features[0])

        _activation_summary(softmax_linear)

    return softmax_linear, last_hidden, h_pool

def imagegate(x,y, size, carry_bias=-1.0):
    # gets two inputs, choosing a bernouli mask between them
  W_Ty = tf.Variable(tf.truncated_normal([size, size], stddev=0.1), name="weight_transform2")
  b_T = tf.Variable(tf.constant(carry_bias, shape=[size]), name="bias_transform")
  T = tf.sigmoid(tf.add(tf.matmul(y, W_Ty),b_T), name="transform_gate")
  C = tf.sub(1.0, T, name="carry_gate")

  z = tf.add(tf.mul(y, T), tf.mul(x, C), "z")

  return z
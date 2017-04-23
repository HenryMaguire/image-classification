import helper as helper
import numpy as np

from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
fitted = False
def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    global fitted # A little hacky but works
    if fitted == False:
        lb.fit(x)
        fitted = True
    return lb.transform(x)

import pickle
import numpy as np
import problem_unittests as tests
import helper

# Load the Preprocessed Validation data
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))
print "done"

import tensorflow as tf

def neural_net_image_input(image_shape):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    batch_shape = [None, image_shape[0], image_shape[1], image_shape[2]]
    x =tf.placeholder(tf.float32, batch_shape, name='x')
    return x


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    batch_shape = [None, n_classes]
    y = tf.placeholder(tf.float32, batch_shape, name='y')
    return y


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    return tf.placeholder(tf.float32, name='keep_prob')


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tf.reset_default_graph()
tests.test_nn_image_inputs(neural_net_image_input)
tests.test_nn_label_inputs(neural_net_label_input)
tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_one_hot_encode(one_hot_encode)
fitted = False

def conv2d(x_tensor, conv_num_outputs, conv_ksize,
                   conv_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool [batch=1, height, width, channels=1]
    :param pool_strides: Stride 2-D Tuple for pool [batch=1, height, width, channels=1]
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    weights_shape = (int(x_tensor.shape[1]), int(x_tensor.shape[2]), int(x_tensor.shape[3]), conv_num_outputs)
    weights = tf.Variable(tf.truncated_normal(weights_shape))
    biases = tf.Variable(tf.zeros(conv_num_outputs))
    conv_strides=[1, conv_strides[0], conv_strides[1], 1]
    conv = tf.nn.conv2d(x_tensor, weights, strides=conv_strides, padding='SAME')
    conv = tf.nn.bias_add(conv, biases)
    conv = tf.nn.relu(conv)
    return conv

def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize,
                   conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool [batch=1, height, width, channels=1]
    :param pool_strides: Stride 2-D Tuple for pool [batch=1, height, width, channels=1]
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    weights_shape = (int(x_tensor.shape[1]), int(x_tensor.shape[2]), int(x_tensor.shape[3]), conv_num_outputs)
    weights = tf.Variable(tf.truncated_normal(weights_shape, mean=0.0, stddev=0.1))
    biases = tf.Variable(tf.zeros(conv_num_outputs))
    conv_strides=[1, conv_strides[0], conv_strides[1], 1]
    conv = tf.nn.conv2d(x_tensor, weights, strides=conv_strides, padding='SAME')
    conv = tf.nn.bias_add(conv, biases)
    conv = tf.nn.relu(conv)
    pool_ksize = [1, pool_ksize[0], pool_ksize[1], 1]
    pool_strides=[1, pool_strides[0], pool_strides[1], 1]
    conv = tf.nn.max_pool(conv,
                          ksize=pool_ksize,
                          strides=pool_strides,
                          padding='SAME')

    return conv


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_con_pool(conv2d_maxpool)
def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    x_shape = x_tensor.shape
    flat_length = int(x_shape[1])*int(x_shape[2])*int(x_shape[3])
    flat_shape = [-1, flat_length] # -1 flattens it to a 1d array
    return tf.reshape(x_tensor, flat_shape)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_flatten(flatten)

def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    input_size = x_tensor.get_shape().as_list()[1]
    weights = tf.Variable(tf.truncated_normal([input_size, num_outputs], mean=0.0, stddev=0.1))
    biases = tf.Variable(tf.zeros(num_outputs))
    fc = tf.add(tf.matmul(x_tensor, weights), biases)
    return tf.nn.relu(fc)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_fully_conn(fully_conn)

def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    input_size = x_tensor.get_shape().as_list()[1]
    weights = tf.Variable(tf.truncated_normal([input_size, num_outputs], mean=0.0, stddev=0.1))
    biases = tf.Variable(tf.zeros(num_outputs))
    fc = tf.add(tf.matmul(x_tensor, weights), biases)
    return fc


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_output(output)

reload(tests)
def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    optimize = session.run(optimizer, feed_dict={
                x: feature_batch,
                y: label_batch,
                keep_prob: keep_probability})
    return optimize

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
#tests.test_train_nn(train_neural_network)

def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # TODO: Apply 1, 2, or 3 Convolution and Max Pool layers
    #    Play around with different number of outputs, kernel size and stride
    # Function Definition from Above:
    #    conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
    conv_layer1 = conv2d_maxpool(x, 32, (3,3),
                   (1,1), (2,2), (2,2))
    #conv_layer2 = conv2d_maxpool(conv_layer1, 32, (3,3),
    #               (1,1),(2,2),(2,2))
    #dropout = tf.nn.dropout(conv_layer2, keep_prob)
    conv_layerf = conv2d_maxpool(conv_layer1, 64, (3,3),
                                 (1,1), (2,2), (2,2))

    #conv_layer4 = (conv_layer2, 128, (5,5),
    #               (2,2))
    # TODO: Apply a Flatten Layer
    # Function Definition from Above:
    #   flatten(x_tensor)
    flatten_layer = flatten(conv_layerf)


    # TODO: Apply 1, 2, or 3 Fully Connected Layers
    #    Play around with different number of outputs
    # Function Definition from Above:
    #   fully_conn(x_tensor, num_outputs)
    fc_layer1 = fully_conn(flatten_layer, 256)
    #dropout = tf.nn.dropout(fc_layer1, keep_prob)
    fc_layerf = fully_conn(fc_layer1, 128)
    # TODO: Apply an Output Layer
    #    Set this to the number of classes
    # Function Definition from Above:
    #   output(x_tensor, num_outputs)
    output_layer = output(fc_layerf, len(lb.classes_))
    return output_layer


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""

##############################
## Build the Neural Network ##
##############################

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

tests.test_conv_net(conv_net)

def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    loss = sess.run(cost, feed_dict={
                x: feature_batch,
                y: label_batch,
                   keep_prob: 1.})
    valid_acc = sess.run(accuracy, feed_dict={
                x: valid_features,
                y: valid_labels,
                keep_prob: 1.})
    print "Loss: {}  |  Validation accuracy: {}".format(loss, valid_acc)
    return loss, valid_acc


# TODO: Tune Parameters
epochs = 50
batch_size = 64
keep_probability = 1.

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
print('Checking the Training on a Single Batch...')
with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2)) as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(epochs):
        batch_i = 1
        for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
            train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
        print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i))
        print_stats(sess, batch_features, batch_labels, cost, accuracy)


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
save_model_path = './image_classification'

print('Training...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            print 'Epoch {}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i)
            print_stats(sess, batch_features, batch_labels, cost, accuracy)

    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)

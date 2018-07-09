
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)



image_size = 28
num_labels = 10
num_channels = 1 # grayscale

import numpy as np

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])



batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size,image_size,image_size,num_channels))
    tf_train_labels = tf.placeholder(tf.float32,shape=(batch_size,num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    layer1_weights = tf.Variable(tf.truncated_normal([patch_size,patch_size,num_channels,depth],stddev = 0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))
    layer2_weights = tf.Variable(tf.truncated_normal([patch_size,patch_size,depth,depth],stddev = 0.1))
    layer2_biases = tf.Variable(tf.constant(1.0,shape=[depth]))
    layer3_weights = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, num_hidden],stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0,shape=[num_hidden]))
    layer4_weights = tf.Variable(tf.truncated_normal([num_hidden,num_labels],stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0,shape=[num_labels]))


    # def model(data):
    #     conv = tf.nn.conv2d(data,layer1_weights,[1,2,2,1],padding='SAME')
    #     hidden = tf.nn.relu(conv+layer1_biases)
    #     conv = tf.nn.conv2d(hidden,layer2_weights,[1,2,2,1],padding='SAME')
    #     hidden = tf.nn.relu(conv+layer2_biases)
    #     shape = hidden.get_shape().as_list()
    #     reshape = tf.reshape(hidden,[shape[0],shape[1]*shape[2]*shape[3]])
    #     hidden = tf.nn.relu(tf.matmul(reshape,layer3_weights)+layer3_biases)
    #     return tf.matmul(hidden,layer4_weights)+layer4_biases
    def model(data):
        conv1 = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
        hidden1 = tf.nn.relu(conv1 + layer1_biases)
        conv2 = tf.nn.conv2d(hidden1, layer2_weights, [1, 2, 2, 1], padding='SAME')
        hidden2 = tf.nn.relu(conv2 + layer2_biases)
        shape1 = hidden2.get_shape().as_list()
        reshape2 = tf.reshape(hidden2, [shape1[0], shape1[1] * shape1[2] * shape1[3]])
        hidden3 = tf.nn.relu(tf.matmul(reshape2, layer3_weights) + layer3_biases)
        return conv1, hidden1, conv2,hidden2,shape1, reshape2, hidden3,tf.matmul(hidden3, layer4_weights) + layer4_biases

    conv1, hidden1, conv2,hidden2,shape1, reshape2, hidden3,logits_train = model(tf_train_dataset)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits_train))
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,logits=logits))

    optimizer =tf.train.GradientDescentOptimizer(0.05).minimize(loss)
    train_prediction = tf.nn.softmax(logits_train)
    conv1, hidden1, conv2,hidden2,shape1, reshape2, hidden3,logits_valid = model(tf_valid_dataset)
    valid_prediction = tf.nn.softmax(logits_valid)
    conv1, hidden1, conv2,hidden2,shape1, reshape2, hidden3,logits_test = model(tf_test_dataset)
    test_prediction = tf.nn.softmax(logits_test)


num_steps = 1001

with tf.Session(graph= graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step*batch_size)%(train_labels.shape[0]-batch_size)
        batch_data = train_dataset[offset:(offset+batch_size),:,:,:]
        print(batch_data.shape)
        batch_labels = train_labels[offset:(offset+batch_size),:]
        print(batch_labels.shape)
        feed_dict ={tf_train_dataset:batch_data,tf_train_labels:batch_labels}
        _,l,predictions,layer_result1_weights, layer_result1_biases, layer_result2_weights, layer_result2_biases, layer_result3_weights, layer_result3_biases, layer_result4_weights, layer_result4_biases,conv1_eval,hidden1_eval, conv2_eval,hidden2_eval, reshape2_eval, hidden3_eval,logits_eval = session.run([optimizer,loss,train_prediction,layer1_weights, layer1_biases, layer2_weights, layer2_biases, layer3_weights, layer3_biases, layer4_weights, layer4_biases,conv1, hidden1, conv2,hidden2, reshape2, hidden3,logits_train],feed_dict=feed_dict)
        if (step % 100 == 0):
          print('Minibatch loss at step %d: %f' % (step, l))
          print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
          #print(layer_result1_weights, layer_result1_biases, layer_result2_weights, layer_result2_biases, layer_result3_weights, layer_result3_biases, layer_result4_weights, layer_result4_biases)
          print("layer_result1_weights.shape","layer_result1_biases.shape","layer_result2_weights.shape","layer_result2_biases.shape","layer_result3_weights.shape","layer_result3_biases.shape","layer_result4_weights.shape","layer_result4_biases")
          print(layer_result1_weights.shape, layer_result1_biases.shape, layer_result2_weights.shape, layer_result2_biases.shape, layer_result3_weights.shape, layer_result3_biases.shape, layer_result4_weights.shape, layer_result4_biases)
          print("conv1_eval.shape","hidden1_eval.shape","conv2_eval.shape,hidden2_eval.shape","reshape2_eval.shape","hidden3_eval.shape,logits_eval.shape")
          print(conv1_eval.shape, hidden1_eval.shape, conv2_eval.shape,hidden2_eval.shape, reshape2_eval.shape, hidden3_eval.shape,logits_eval.shape)
          print('Validation accuracy: %.1f%%' % accuracy(
            valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

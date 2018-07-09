
from __future__ import print_function

import tensorflow as tf

#import mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

#parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

#network_parameters
n_input = 784
n_classes = 10
dropout = 0.75

#tf_graph_inputs
x = tf.placeholder(tf.float32,[None,n_input])
y = tf.placeholder(tf.float32,[None,n_classes])
keep_prob = tf.placeholder(tf.float32)

def conv2d(x,W,b,strides=1):
    x = tf.nn.conv2d(x,W,[1,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

def maxpool(x,k=2):
    return tf.nn.max_pool(x, ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

#model_creation
def conv_net(x,weights,biases,dropout):
    x = tf.reshape(x,shape=[-1,28,28,1])
    conv1 = conv2d(x,weights['wc1'],biases['bc1'])
    conv1 = maxpool(conv1)
    conv2 = conv2d(conv1,weights['wc2'],biases['bc2'])
    conv2 = maxpool(conv2)
    conv2 = tf.reshape(conv2,shape=[-1,7*7*64])
    fc1 = tf.add(tf.matmul(conv2,weights['wd1']),biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1,dropout)
    out = tf.add(tf.matmul(fc1,weights['out']),biases['out'])
    return out




weights ={
         #5*5 conv nets 1 input , 32 output
         'wc1':tf.Variable(tf.random_normal([5,5,1,32])),
         #5*5 conv nets 32 input , 64 output
         'wc2':tf.Variable(tf.random_normal([5,5,32,64])),
         #fully connected layer 7*7*64 inputs 1024 outputs
         'wd1':tf.Variable(tf.random_normal([7*7*64,1024])),
         #output 1024 inputs num_classes output
         'out': tf.Variable(tf.random_normal([1024,n_classes]))
}
biases = {
         'bc1':tf.Variable(tf.random_normal([32])),
         #5*5 conv nets 32 input , 64 output
         'bc2':tf.Variable(tf.random_normal([64])),
         #fully connected layer 7*7*64 inputs 1024 outputs
         'bd1':tf.Variable(tf.random_normal([1024])),
         #output 1024 inputs num_classes output
         'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = conv_net(x,weights,biases,keep_prob)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,logits=logits))

optimizer =tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(pred,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 1
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
              train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
              print('step %d, training accuracy %g' % (i, train_accuracy))
            optimizer.run(feed_dict={x: batch[0], y: batch[1], keep_prob: dropout})
        print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))

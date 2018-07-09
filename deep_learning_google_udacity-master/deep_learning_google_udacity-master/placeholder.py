import tensorflow as tf
import numpy as np
x = tf.placeholder(tf.float32, [None,1024])
y = tf.matmul(x, x)

with tf.Session() as sess:
  rand_array = np.random.rand(1024)
  print(sess.run(y, feed_dict={x: rand_array}))

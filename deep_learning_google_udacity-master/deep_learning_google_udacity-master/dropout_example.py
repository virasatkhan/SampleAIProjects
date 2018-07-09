### imports
import tensorflow as tf

### constant data
x  = [[0.,0.],[1.,1.],[1.,0.],[0.,1.]]
y_ = [[1.,0.],[1.,0.],[0.,1.],[0.,1.]]

### induction

# Layer 0 = the x2 inputs
x0 = tf.constant( x  , dtype=tf.float32 )
y0 = tf.constant( y_ , dtype=tf.float32 )

keep_prob = tf.placeholder( dtype=tf.float32 )

# Layer 1 = the 2x12 hidden sigmoid
m1 = tf.Variable( tf.random_uniform( [2,12] , minval=0.1 , maxval=0.9 , dtype=tf.float32  ))
b1 = tf.Variable( tf.random_uniform( [12]   , minval=0.1 , maxval=0.9 , dtype=tf.float32  ))


########## DROP CONNECT
# - use this to preform "DropConnect" flavor of dropout
dropConnect = tf.nn.dropout( m1, keep_prob ) * keep_prob
h1 = tf.sigmoid( tf.matmul( x0, dropConnect ) + b1 )

########## DROP OUT
# - uncomment this to use "regular" dropout
#h1 = tf.nn.dropout( tf.sigmoid( tf.matmul( x0,m1 ) + b1 ) , keep_prob )


# Layer 2 = the 12x2 softmax output
m2 = tf.Variable( tf.random_uniform( [12,2] , minval=0.1 , maxval=0.9 , dtype=tf.float32  ))
b2 = tf.Variable( tf.random_uniform( [2]   , minval=0.1 , maxval=0.9 , dtype=tf.float32  ))
y_out = tf.nn.softmax( tf.matmul( h1,m2 ) + b2 )


# loss : sum of the squares of y0 - y_out
loss = tf.reduce_sum( tf.square( y0 - y_out ) )

# training step : gradient decent (1.0) to minimize loss
train = tf.train.AdamOptimizer(1e-2).minimize(loss)

### training
# run 5000 times using all the X and Y
# print out the loss and any other interesting info
with tf.Session() as sess:
  sess.run( tf.initialize_all_variables() )
  print "\nloss"
  for step in range(5000) :
    sess.run(train,feed_dict={keep_prob:0.5})
    if (step + 1) % 100 == 0 :
      print sess.run(loss,feed_dict={keep_prob:1.})


  results = sess.run([m1,b1,m2,b2,y_out,loss],feed_dict={keep_prob:1.})
  labels  = "m1,b1,m2,b2,y_out,loss".split(",")
  for label,result in zip(*(labels,results)) :
    print ""
    print label
    print result

print ""

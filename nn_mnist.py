import gzip
import cPickle

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
import time


def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x, train_y = train_set
train_y = one_hot(train_y.astype(int), 10)
valid_x, valid_y = valid_set
valid_y = one_hot(valid_y.astype(int), 10)
test_x, test_y = test_set
test_y = one_hot(test_y.astype(int), 10)

# ---------------- Visualizing some element of the MNIST dataset --------------

#import matplotlib.cm as cm
#import matplotlib.pyplot as plt

#plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
#plt.show()  # Let's see a sample
#print train_y[57]


# TODO: the neural net!!
x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

# input -> inner nn layer

# 28*28 entries (each pixel of the image) and 10 neurons
W1 = tf.Variable(np.float32(np.random.rand(784, 10)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

#OPT - ANOTHER NN LAYER
W1h = tf.Variable(np.float32(np.random.rand(10, 10)) * 0.1)
b1h = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

# inner nn layer -> outter nn layer

# 10 entries and 10 neurons
W2 = tf.Variable(np.float32(np.random.rand(10, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

# outter nn layer -> output (10 neurons codify our number using a 10-digit-onehot.code 1 = 0000000001, 2 = 0000000010, ...
h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
#h = tf.matmul(x, W1) + b1  # Try this!
h2 = tf.matmul(h, W1h) + b1h
y = tf.nn.softmax(tf.matmul(h2, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

#Variables used in our train-and-validate loop
batch_size = 20
evolErrorPrev = 10000
evolErrorAct = 10000
epoch = 0

#We are using a while loop so we can assume that when we stop its because we reached the accuracy threshold of our nn
while evolErrorPrev >= evolErrorAct:
    for jj in xrange(len(train_x) / batch_size):
        # Training zone
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
        # End of training zone

    # Validation zone
    evolErrorPrev = evolErrorAct
    evolErrorAct = sess.run(loss, feed_dict={x: valid_x, y_: valid_y})
    print "Epoch #:", epoch, "Error: ", evolErrorAct
    result = sess.run(y, feed_dict={x: valid_x})
    epoch = epoch + 1
    #for b, r in zip(y_data_valid, result):
    #    print b, "-->", r
    print "----------------------------------------------------------------------------------"
    # End of validation zone
# Once we are done validating our nn we test it using the other array counting how many errors it makes.
error = 0

# We finally test our nn using the test dataset and compare it with the real results
result = sess.run(y, feed_dict={x: test_x})
for b, r in zip(test_y, result):
    if np.argmax(b) != np.argmax(r):
        error += 1
    #print b, "-->", r
print "----------------------------------------------------------------------------------"
print "Error:", error

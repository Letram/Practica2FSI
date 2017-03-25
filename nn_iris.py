import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
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


data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data

x_data = data[:, 0:4].astype('f4')  # the samples are the four first rows of data
y_data = one_hot(data[:, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code

x_data_train = x_data[:105]
y_data_train = y_data[:105]

x_data_valid = x_data[105:128]
y_data_valid = y_data[105:128]

x_data_test = x_data[128:]
y_data_test = y_data[128:]

print "\nSome samples..."
for i in range(20):
    print x_data[i], " -> ", y_data[i]
print

x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

# input -> inner nn layer

# 4 entries and 5 neurons
W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

#OPT ANOTHER NN LAYER
W1h = tf.Variable(np.float32(np.random.rand(5, 5)) * 0.1)
b1h = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

# inner nn layer -> outter nn layer

# 5 entries and 3 neurons
W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

# outter nn layer -> output
#h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
h = tf.matmul(x, W1) + b1  # Try this!
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

batch_size = 20

for epoch in xrange(150):
    for jj in xrange(len(x_data_train) / batch_size):
        # Training zone
        batch_xs = x_data_train[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data_train[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
        # End of training zone

    # Validation zone
    print "Epoch #:", epoch, "Error: ", sess.run(loss, feed_dict={x: x_data_valid, y_: y_data_valid})
    result = sess.run(y, feed_dict={x: x_data_valid})
    #for b, r in zip(y_data_valid, result):
    #    print b, "-->", r
    print "----------------------------------------------------------------------------------"
    # End of validation zone

# Once we are done validating our nn we test it using the other array counting how many errors it makes.
error = 0

# We finally test our nn using the test dataset and compare it with the real results
result = sess.run(y, feed_dict={x: x_data_test})
for b, r in zip(y_data_test, result):
    if np.argmax(b) != np.argmax(r):
        error += 1
    print b, "-->", r
print "----------------------------------------------------------------------------------"
print "Error:", error
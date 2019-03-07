#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import scipy
import random
from scipy import io
from scipy.io import wavfile
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


x = tf.placeholder(tf.float32, shape=[None, 784])
t = tf.placeholder(tf.float32, shape=[None, 10])
keep_drop = 0.8

W1 = tf.Variable(tf.truncated_normal([784, 200], stddev=0.1))
b1 = tf.Variable(tf.zeros([200]))
z1 = tf.matmul(x, W1) + b1
h1 = tf.nn.tanh(z1)
h1_drop = tf.nn.dropout(h1, keep_drop)

W2 = tf.Variable(tf.truncated_normal([200, 200], stddev=0.1))
b2 = tf.Variable(tf.zeros([200]))
z2 = tf.matmul(h1_drop, W2) + b2
h2 = tf.nn.relu(z2)
h2_drop = tf.nn.dropout(h2, keep_drop)

W3 = tf.Variable(tf.truncated_normal([200, 200], stddev=0.1))
b3 = tf.Variable(tf.zeros([200]))
z3 = tf.matmul(h2_drop, W3) + b3
h3 = tf.nn.relu(z3)
h3_drop = tf.nn.dropout(h3, keep_drop)

W4 = tf.Variable(tf.truncated_normal([200, 200], stddev=0.1))
b4 = tf.Variable(tf.zeros([200]))
z4 = tf.matmul(h3_drop, W4) + b4
h4 = tf.nn.relu(z4)
h4_drop = tf.nn.dropout(h4, keep_drop)

W5 = tf.Variable(tf.truncated_normal([200, 200], stddev=0.1))
b5 = tf.Variable(tf.zeros([200]))
z5 = tf.matmul(h4_drop, W5) + b5
h5 = tf.nn.relu(z5)
h5_drop = tf.nn.dropout(h5, keep_drop)

V = tf.Variable(tf.truncated_normal([200, 10], stddev=0.1))
c = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(h5_drop, V) + c)

delta = 1e-35
cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y+delta), axis=1))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

##以降sessionスタート
train_size = 0.8
mnist = scipy.io.loadmat("/home/ec2-user/scikit_learn_data/mnist-original.mat")
X = mnist['data'].T
Y = np.eye(10)[(mnist['label'][0].T).astype(int)]

X_train, X_test, Y_train, Y_test =\
    train_test_split(X, Y, train_size=train_size)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

epochs = 10
N = 70000
batch_size = 200
n_batches = (int)(N * train_size) // batch_size
for epoch in range(epochs):
    X_, Y_ = shuffle(X_train, Y_train)
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        loss, acc, _ = sess.run([cross_entropy, accuracy, train_step], feed_dict={x: X_[start:end], t: Y_[start:end]})
    print('epoch:', epoch, ' loss:', loss, ' accuracy:', acc)
accuracy_rate = accuracy.eval(session=sess, feed_dict={x: X_test, t: Y_test})
print('accuracy: ', accuracy_rate)


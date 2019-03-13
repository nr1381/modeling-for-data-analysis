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
t = tf.placeholder(tf.float32, shape=[None, 784])

W1 = tf.Variable(tf.truncated_normal([784, 200], stddev=0.0001))
b1 = tf.Variable(tf.zeros([200]))
z1 = tf.matmul(x, W1) + b1
h1 = tf.nn.relu(z1)

V = tf.Variable(tf.truncated_normal([200, 784], stddev=0.0001))
c = tf.Variable(tf.zeros([784]))
y = tf.matmul(h1, V) + c

mean_square = tf.reduce_mean(tf.square(y - t))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(mean_square)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

##以降sessionスタート
file_path = "/home/ec2-user/develop/prml/src/auto_encorder/result.txt"
train_size = 0.8
mnist = scipy.io.loadmat("/home/ec2-user/scikit_learn_data/mnist-original.mat")
X = mnist['data'].T
Y = mnist['data'].T

X_train, X_test, Y_train, Y_test =\
    train_test_split(X, Y, train_size=train_size)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

epochs = 1000
N = 70000
batch_size = 200
n_batches = (int)(N * train_size) // batch_size
for epoch in range(epochs):
    X_, Y_ = shuffle(X_train, Y_train)
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        loss, acc, _, h11 = sess.run([mean_square, accuracy, train_step, y], feed_dict={x: X_[start:end], t: Y_[start:end]})
    print('epoch:', epoch, ' loss:', loss, ' accuracy:', acc)
accuracy_rate = accuracy.eval(session=sess, feed_dict={x: X_test, t: Y_test})
print('accuracy: ', accuracy_rate)
print(type(h11))
np.savetxt('test_10000.csv', h11, delimiter=',')


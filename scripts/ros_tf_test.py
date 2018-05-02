#!/usr/bin/env python

import rospy
import numpy as np
import time
from std_msgs.msg import String

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class RosTFTest:
  def __init__(self):
    self.bridge = CvBridge()
    self.x = tf.placeholder(tf.float32, [None, 784])
    self.W_conv1 = self.weight_variable([5, 5, 1, 32])
    self.b_conv1 = self.bias_variable([32])

    self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])

    self.h_conv1 = tf.nn.relu(self.conv2d(self.x_image, self.W_conv1) + self.b_conv1)
    self.h_pool1 = self.max_pool_2x2(self.h_conv1)

    self.W_conv2 = self.weight_variable([5, 5, 32, 64])
    self.b_conv2 = self.bias_variable([64])
    self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
    self.h_pool2 = self.max_pool_2x2(self.h_conv2)

    self.W_fc1 = self.weight_variable([7*7*64, 1024])
    self.b_fc1 = self.bias_variable([1024])
    self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7*7*64])
    self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)

    self.keep_prob = tf.placeholder(tf.float32)
    self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

    self.W_fc2 = self.weight_variable([1024, 10])
    self.b_fc2 = self.bias_variable([10])

    self.y_conv = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2

    self.sess = tf.Session()

  def excute(self):
    pub = rospy.Publisher('/image', Image, queue_size=10)
    sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.ImageCallback)
    rospy.init_node('input', anonymous=True)
    r = rospy.Rate(10) # 10hz

    init = tf.initialize_all_variables()

    self.sess.run(init)

    saver = tf.train.Saver()
    print "Load Model"
    saver.restore(self.sess, "./model/model.ckpt")

    while not rospy.is_shutdown():
      r.sleep()

  def ImageCallback(self, data):
    try:
      # sensor_msgs::Image to numpy.ndarray
      cv_image = self.bridge.imgmsg_to_cv2(data, "mono8")
      cv_image = cv2.resize(cv_image, (28, 28))
      # for input
      ximage = cv_image.flatten().astype(np.float32)/255.0
      # calcurate
      prediction = np.argmax(self.y_conv.eval(session=self.sess, feed_dict={self.x:[ximage], self.keep_prob:1.0})[0])
      print(prediction)
    except CvBridgeError as e:
      print(e)

  def train(self, loop=20000):
    rospy.init_node('learn_and_save', anonymous=True)
    r = rospy.Rate(10)

    start_time = time.time()
    print "start:" + str(start_time)

    print "--- Loading ---"
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    print "--- Finished ---"

    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_conv, labels=y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.initialize_all_variables()

    self.sess.run(init)

    print " --- Start Training ---"
    for i in range(loop):
      batch_xs, batch_ys = mnist.train.next_batch(50)
      if i%100 == 0:
        print("i=" + str(i) + "/" + str(loop))
        print(self.sess.run(accuracy, feed_dict={self.x:mnist.test.images, y_:mnist.test.labels, self.keep_prob:1.0}))
      self.sess.run(train_step, feed_dict={self.x:batch_xs, y_:batch_ys, self.keep_prob:0.5})
      if rospy.is_shutdown():
        exit()

    print " --- Finished ---"

    print "Accuracy"
    print(self.sess.run(accuracy, feed_dict={self.x:mnist.test.images, y_:mnist.test.labels, self.keep_prob:0.5}))

    end_time = time.time()
    print "End time" + str(end_time)
    print "time:" + str(end_time - start_time)
    saver = tf.train.Saver()
    saver.save(self.sess, "./model/model.ckpt")

    while not rospy.is_shutdown():
      r.sleep()

  def continue_learning(self, loop=20000):
    rospy.init_node('continue_learning', anonymous=True)
    r = rospy.Rate(10)

    start_time = time.time()
    print "start:" + str(start_time)

    print "--- Loading ---"
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    print "--- Finished ---"

    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_conv, labels=y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.initialize_all_variables()

    self.sess.run(init)

    saver = tf.train.Saver()
    print "--- Load Model ---"
    saver.restore(self.sess, "./model/model.ckpt")

    print "--- Start Training ---"
    for i in range(loop):
      batch_xs, batch_ys = mnist.train.next_batch(50)
      if i%100 == 0:
        print("i=" + str(i) + "/" + str(loop))
        print(self.sess.run(accuracy, feed_dict={self.x:mnist.test.images, y_:mnist.test.labels, self.keep_prob:1.0}))
      self.sess.run(train_step, feed_dict={self.x:batch_xs, y_:batch_ys, self.keep_prob:0.5})
      if rospy.is_shutdown():
        exit()

    print "--- finished ---"

    print "Accuracy"
    print(self.sess.run(accuracy, feed_dict={self.x:mnist.test.images, y_:mnist.test.labels, self.keep_prob:0.5}))

    end_time = time.time()
    print "End time:" + str(end_time)
    print "time:" + str(end_time - start_time)

    saver.save(self.sess, "./model/model.ckpt")

    while not rospy.is_shutdown():
      r.sleep()

  def weight_variable(self, shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(self, shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def conv2d(self, x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool_2x2(self, x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


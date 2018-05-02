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
    self.W = tf.Variable(tf.zeros([784, 10]))
    self.b = tf.Variable(tf.zeros([10]))
    self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)
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
      prediction = np.argmax(self.y.eval(session=self.sess, feed_dict={self.x:[ximage]})[0])
      print(prediction)
    except CvBridgeError as e:
      print(e)

  def train(self, loop=1000):
    rospy.init_node('learn_and_save', anonymous=True)
    r = rospy.Rate(10)

    start_time = time.time()
    print "start:" + str(start_time)

    print "--- Loading ---"
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    print "--- Finished ---"

    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = -tf.reduce_sum(y_*tf.log(self.y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    init = tf.initialize_all_variables()

    self.sess.run(init)

    print " --- Start Training ---"
    for i in range(loop):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      self.sess.run(train_step, feed_dict={self.x:batch_xs, y_:batch_ys})
    print " --- Finished ---"

    correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print "Accuracy"
    print(self.sess.run(accuracy, feed_dict={self.x:mnist.test.images, y_:mnist.test.labels}))

    end_time = time.time()
    print "End time" + str(end_time)
    print "time:" + str(end_time - start_time)
    saver = tf.train.Saver()
    saver.save(self.sess, "./model/model.ckpt")

    while not rospy.is_shutdown():
      r.sleep()

  def continue_learning(self, loop=1000):
    rospy.init_node('continue_learning', anonymous=True)
    r = rospy.Rate(10)

    start_time = time.time()
    print "start:" + str(start_time)

    print "--- Loading ---"
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    print "--- Finished ---"

    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = -tf.reduce_sum(y_*tf.log(self.y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    init = tf.initialize_all_variables()

    self.sess.run(init)

    saver = tf.train.Saver()
    print "--- Load Model ---"
    saver.restore(self.sess, "./model/model.ckpt")

    print "--- Start Training ---"
    for i in range(loop):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      self.sess.run(train_step, feed_dict={self.x:batch_xs, y_:batch_ys})
    print "--- finished ---"

    correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print "Accuracy"
    print(self.sess.run(accuracy, feed_dict={self.x:mnist.test.images, y_:mnist.test.labels}))

    end_time = time.time()
    print "End time:" + str(end_time)
    print "time:" + str(end_time - start_time)

    saver.save(self.sess, "./model/model.ckpt")

    while not rospy.is_shutdown():
      r.sleep()




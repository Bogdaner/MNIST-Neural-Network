import tensorflow as tf
import numpy as np
import cv2
from tensorflow.examples.tutorials.mnist import input_data


def load():
    #Custom Input
    INPUTS = 6
    images = np.zeros((INPUTS, 28*28))
    labels = np.zeros((INPUTS, 10))
    i = 0
    for im in [6, 0, 3, 5, 7, 8]:
        img = cv2.imread(str(im) + ".png", cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))

        cv2.imshow(str(im), mat=(cv2.resize(img, (200, 200))))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        flatten = img.flatten() / 255.0
        images[i] = flatten
        correct_value = np.zeros(10)
        correct_value[im] = 1
        labels[i] = correct_value
        i = i + 1

    return images, labels


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def main(_):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    x = tf.placeholder(tf.float32, shape=[None, 28*28], name="input")
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    #First Layer
    W_conv1 = weight_variable([3, 3, 1, 16])
    b_conv1 = bias_variable([16])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    #Second Layer
    W_conv2 = weight_variable([3, 3, 16, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #Fully-connected Layer
    W_fc1 = weight_variable([7 * 7 * 32, 256])  # 7 from application 2x2 max pooling 2 times
    b_fc1 = bias_variable([256])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 32])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    #Dropout
    #keep_prob = tf.placeholder(tf.float32)
    #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #Readout Layer
    W_fc2 = weight_variable([256, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    for i in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(50)
        _, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            print("Loss {}, Accuracy {}".format(loss, train_accuracy))

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    #Custom Input
    images, labels = load()
    prediction = tf.argmax(y_conv, 1)
    print(sess.run(prediction, feed_dict={x: images, y_: labels}))


if __name__ == '__main__':
  tf.app.run()
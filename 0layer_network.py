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


def main(_):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    x = tf.placeholder(tf.float32, shape=[None, 28*28], name="input")
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    W = tf.get_variable("W", shape=[784, 10], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.zeros([10]), name="b")
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    cross_entropy = -tf.reduce_sum(y_*tf.log(y)) # tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Getting pic from mnist
    # img, lab = mnist.train.next_batch(1)
    # img[0] = img[0] * 255
    # tmp_img = img[0].reshape(28, 28)
    # cv2.imwrite("m.png", tmp_img)

    file_loss = open("loss.txt", "w")
    file_acc = open("accuracy.txt", "w")
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        _, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
        train_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        #print("Loss {}, Accuracy {}".format(loss, train_accuracy))

        file_loss.write(str(loss) + "\n")
        file_acc.write(str(train_accuracy) + "\n")

    file_loss.close()
    file_acc.close()
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

    #Custom Input
    images, labels = load()
    prediction = tf.argmax(y, 1)
    print(sess.run(prediction, feed_dict={x: images, y_: labels}))


if __name__ == '__main__':
  tf.app.run()
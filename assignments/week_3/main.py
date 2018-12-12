import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
import numpy as np
import gzip
import matplotlib.pyplot as plt
import time

def display(image, label):
    plt.imshow(image)
    plt.savefig("image.svg")
    print("Digit: {}".format(label))


def load_data():
    with open('../../data/mnist/train-images-idx3-ubyte.gz', 'rb') as f:
        train_images = np.squeeze(extract_images(f))
    with open('../../data/mnist/train-labels-idx1-ubyte.gz', 'rb') as f:
        train_labels = extract_labels(f)
    with open('../../data/mnist/t10k-images-idx3-ubyte.gz', 'rb') as f:
        test_images = np.squeeze(extract_images(f))
    with open('../../data/mnist/t10k-labels-idx1-ubyte.gz', 'rb') as f:
        test_labels = extract_labels(f)
    return train_images, train_labels, test_images, test_labels


def flat(images):
    return images.reshape((images.shape[0],-1))


def one_hot(labels):
    one_hot_labels = np.zeros((labels.shape[0],10))
    for num in range(labels.shape[0]):
        one_hot_labels[num, labels[num]] = 1.0
    return one_hot_labels


class MNIST_CNN:
    def __init__(self, wd_factor, learning_rate):
        self.wd_factor = wd_factor
        self.learning_rate = learning_rate
        self.train_pointer = 0
        self.test_pointer = 0

        self.sess = tf.Session()

        self.input = tf.placeholder(dtype=tf.float32, shape=[None,784], name='input')
        self.ground_truth = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='ground_truth')

        self.is_training = tf.placeholder(tf.bool, name='is_training')
        print(self.input)

        self._build_graph()

    def _build_graph(self):
        weights = []

        with tf.variable_scope('layers'):
            h = tf.layers.conv2d(self.input, 28, (11, 11), strides = (4, 4),
                    padding = 'same', data_format = 'channels_last',
                    activation = None, use_bias = True,
                    kernel_initializer = tf.glorot_uniform_initializer(),
                    name = 'conv1')
            print(h)

            h = tf.layers.batch_normalization(h, training = self.is_training)
            h = tf.nn.relu(h)
            h = tf.layers.conv2d(h, 
            h = tf.layers.dense(h, 196, kernel_initializer=tf.glorot_uniform_initializer(), activation=tf.tanh, name='2')
            print(h)
            h = tf.layers.dense(h, 49, kernel_initializer=tf.glorot_uniform_initializer(), activation=tf.tanh, name='3')
            print(h)
            self.logits = tf.layers.dense(h, 10, kernel_initializer=tf.glorot_uniform_initializer(), activation=tf.identity, name='4')
            print(self.logits)
            self.prediction = tf.nn.softmax(self.logits, name='softmax_prediction')

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,labels=self.ground_truth))
            self.loss += self.weight_decay()

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def weight_decay(self):
        loss = 0
        for v in tf.global_variables():
            if 'Adam' in v.name:
                continue
            elif 'kernal' in v.name:
                loss += self.wd_factor * tf.nn.l2.loss(v)
        print(loss)
        return(loss)

    def train_minibatch(self, samples, labels, batch_size):
        if self.train_pointer + batch_size <- samples.shape[0]:
            samples_minibatch = samples[self.train_pointer: self.train_pointer + batch_size]
            labels_minibatch = labels[self.train_pointer: self.train_pointer + batch_size]
            self.train_pointer += batch_size
        else:
            samples_minibatch = samples[self.train_pointer:]
            labels_minibatch = labels[self.train_pointer:]
            self.train_pointer = 0
        return samples_minibatch, labels_minibatch

    def train(self, train_samples, train_labels, train_batch_size, iteration_steps):
        self.sess.run(tf.global_variables_initializer())

        print('Start Training')
        losses = []
        for i in range(iteration_steps):
            samples, labels = self.train_minibatch(train_samples, train_labels, train_batch_size)
            feed_dict = {self.input: samples, self.ground_truth: labels}
            _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
            if i % 50 == 0:
                print("Minibatch loss at step {}: {}".format(i, loss))
                losses.append([i, loss])
        return losses

    def test_minibatch(self, samples, labels, batch_size):
        if self.test_pointer + batch_size <= samples.shape[0]:
            samples_minibatch = samples[self.test_pointer: self.test_pointer + batch_size]
            labels_minibatch = labels[self.test_pointer: self.test_pointer + batch_size]
            self.test_pointer += batch_size
            end_of_epoch = False
        else:
            samples_minibatch = samples[self.test_pointer:]
            labels_minibatch = labels[self.test_pointer:]
            self.test_pointer = 0
            end_of_epoch = True
        return samples_minibatch, labels_minibatch, end_of_epoch

    def test(self, test_samples, test_labels, test_batch_size):
        end_of_eposh = False
        losses = []
        while not end_of_epoch:
            samples, labels, end_of_epoch = self.test_minibatch(test_samples, test_labels, test_batch_size)
            feed_dict = {self.input: samples, self.ground_truth: labels}
            losses.append(self.sess.run(self.loss, feed_dict=feed_dict))
        print("Average test loss: {}".format(np.mean(losses)))


if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = load_data()
    
    train_images_flat = flat(train_images)
    test_images_flat = flat(test_images)
    train_labels_one_hot = one_hot(train_labels)
    test_labels_one_hot = one_hot(test_labels)

    n_train = train_labels.shape[0]
    n_test = test_labels.shape[0]

    #Inspect the data
    example = np.random.choice(np.arange(n_train))
    display(train_images[example],train_labels[example])
    print(train_labels_one_hot[example])

    WD_FACTOR = 0.0001
    LEARNING_RATE = 0.001
    model = MNIST_MLP(WD_FACTOR, LEARNING_RATE)

    TRAIN_BATCH_SIZE = 128
    ITERATIONS = 1000

    start_time = time.time()

    losses = model.train(train_images_flat, train_labels_one_hot, TRAIN_BATCH_SIZE, ITERATIONS)

    end_time = time.time()
    print("Training time: {}s".format(end_time - start_time))

    losses = np.array(losses)
    print(losses.shape)

    iterations = losses[:,0]
    train_loss = losses[:,1]
    plt.figure(figsize=(10,5))
    plt.plot(iterations, train_loss)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training curve")
    plt.show()
    plt.savefig("training_curve.eps")

    TEST_BATCH_SIZE = 128

    model.test(test_images_flat, test_labels_one_hot, TEST_BATCH_SIZE)

    example = np.random.choice(np.arange(n_test))
    
    sample = np.expand_dims(test_images_flat[example], axis=0)
    label = np.expand_dims(test_labels_one_hot[example], axis=0)

    digit = np.where(label[0]==1.0)[0][0]

    feed_dict = {model.input: sample, model.ground_truth: label}
    prediction = model.sess.run(model.prediction, feed_dict)[0]

    image = np.reshape(sample, (32,32))

    print("Test sample digit: {}".format(digit))
    print("Prediction: {}".format(prediction))

    model.sess.close()

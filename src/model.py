import numpy as np
import tensorflow as tf

class NodeSkipGram(object):
    def __init__(self, param):
        self.tensor_graph = tf.Graph()
        self.embedding_size = param["embedding_size"]
        self.layer_size = param["layer_size"]

        self.batch_size = param["batch_size"]
        self.num_node = param["num_node"]
        self.learn_rate = param["learn_rate"]

        with self.tensor_graph.as_default():
            self.x_nodes = tf.placeholder(
                    tf.int32, shape = [None, 1])
            self.delta_ = tf.placeholder(tf.float32,
                    shape = [None, self.num_node])

            self.embeddings = tf.Variable(tf.random_uniform(
                    [self.num_node, self.embedding_size], -1.0, 1.0))

            self.w1 = tf.Variable(tf.random_uniform(
                [self.embedding_size, self.num_node], -1.0, 1.0))
            self.w2 = tf.Variable(tf.random_uniform(
                [self.num_node, self.layer_size], -1.0, 1.0))
            self.b2 = tf.Variable(tf.zeros([self.layer_size]))
            self.w3 = tf.Variable(tf.random_uniform(
                [self.layer_size, self.num_node], -1.0, 1.0))
            self.b3 = tf.Variable(tf.zeros([self.num_node]))

            self.embed_pre = tf.nn.embedding_lookup(self.embeddings, self.x_nodes)

            self.embed = tf.reshape(self.embed_pre, [-1, self.embedding_size])

            self.coefficient = tf.matmul(self.embed, self.w1)

            self.embedding_out = tf.nn.sigmoid(self.coefficient)

            self.layer2_out = tf.nn.sigmoid(
                tf.matmul(self.embedding_out, self.w2) + self.b2)
            self.out = tf.nn.sigmoid(
                tf.matmul(self.layer2_out, self.w3) + self.b3)
            #self.cross_entropy = tf.reduce_mean(
            #       -tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))

            self.mean_quare = tf.reduce_mean(tf.reduce_sum(
                tf.square(self.delta_ - self.out), reduction_indices = [1]))

            self.train_step = tf.train.AdamOptimizer(
                    self.learn_rate).minimize(self.mean_quare)
            #self.train_step = tf.train.AdamOptimizer(
            #       self.learn_rate).minimize(self.cross_entropy)
            #self.train_step = tf.train.GradientDescentOptimizer(
            #       self.learn_rate).minimize(self.cross_entropy)            

    def Train(self, get_batch, epoch_num = 1001):
        with self.tensor_graph.as_default():
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for i in xrange(epoch_num):
                    batch_nodes, batch_y = get_batch(self.batch_size)
                    self.train_step.run({self.x_nodes: batch_nodes,
                        self.delta_: batch_y})
                    if (i % 100 == 0):
                        print (self.mean_quare.eval({
                            self.x_nodes: batch_nodes,
                            self.delta_: batch_y}))
                return sess.run(self.embeddings), sess.run(tf.matmul(self.embeddings, self.w1))

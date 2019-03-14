import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from layers import *
import utils

class Model(object):
    def __init__(self, config, embedding_matrix):
        # self.word_cell = config.word_cell
        self.word_output_size = config.word_output_size
        self.classes = config.classes
        self.aspnum = config.aspnum
        self.max_grad_norm = config.max_grad_norm
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.dropout_keep_proba = config.dropout_keep_proba
        self.lr = config.lr
        self.seed = config.seed
        # self.seed = None
        self.attRandomBase = config.attRandomBase
        self.biRandomBase = config.biRandomBase
        self.aspRandomBase = config.aspRandomBase
        self.Winit = tf.random_uniform_initializer(minval=-1.0*config.attRandomBase, maxval=config.attRandomBase, seed=self.seed)
        # self.Winit = None
        # self.Winit = tf.truncated_normal_initializer(seed=self.seed)
        self.word_cell = tf.contrib.rnn.LSTMCell
        # self.word_cell = tf.contrib.rnn.GRUCell
        with tf.variable_scope('tcm') as scope:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.embedding_matrix = tf.placeholder(shape=(1, self.embedding_size), dtype=tf.float32, name='embedding_matrix')#随便初始化
            if embedding_matrix is None:
                self.embedding_matrix = tf.placeholder(shape=(None, None), dtype=tf.float32, name='embedding_matrix')
                self.embedding_C = tf.placeholder(shape=(None, None), dtype=tf.float32, name='embedding_C')
            else:
                self.embedding_matrix = tf.Variable(initial_value=embedding_matrix, name='embedding_matrix', dtype=tf.float32, trainable=True)
                # # self.position_embedding = tf.Variable(tf.random_uniform(shape=[self.word_output_size * 2], minval=-1.0 * self.aspRandomBase, maxval=self.aspRandomBase, seed=self.seed),\
                # name = 'position_embedding', dtype = tf.float32, trainable = True)

            # self.aspect_embedding_c = tf.Variable(tf.random_uniform(shape=[self.aspnum, self.embedding_size], minval=-1.0 * self.aspRandomBase, maxval=self.aspRandomBase, seed=self.seed),
            #                                   name='aspect_embedding_c', dtype=tf.float32, trainable=True)
            # self.aspect_embedding = tf.Variable(initial_value=asp_embedding_matrix, name='asp_embedding_matrix',
            #                                     dtype=tf.float32, trainable=True)
            # self.context_vector = tf.Variable(tf.truncated_normal(shape=[self.word_output_size * 2]),
            #                                   name='attention_context_vector', dtype=tf.float32, trainable=True)
            # self.aspect_embedding = tf.Variable(tf.truncated_normal(shape=[5, self.embedding_size]),
            #                                   name='aspect_embedding', dtype=tf.float32, trainable=True)

            self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')
            # [document x word]
            self.inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='inputs')
            self.positions = tf.placeholder(shape=(None, None), dtype=tf.int32, name='positions')
            self.inputs_len = tf.placeholder(shape=(None,), dtype=tf.float32, name='inputs')
            self.targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='targets')
            self.textwm = tf.placeholder(shape=(None, None), dtype=tf.float32, name='textwordmask')
            self.targetwm = tf.placeholder(shape=(None, None), dtype=tf.int32, name='targetwordmask')
            self.posmask = tf.placeholder(shape=(None, None), dtype=tf.float32, name='positionmask')
            self.posweight = tf.placeholder(shape=(None, None), dtype=tf.float32, name='positionwei')
            self.text_word_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='text_word_lengths')
            self.target_word_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='target_word_lengths')
            self.labels = tf.placeholder(shape=(None,), dtype=tf.int32, name='labels')
            self.category = tf.placeholder(shape=(None,None), dtype=tf.int32, name='category')

        with tf.variable_scope('embedding'):
            with tf.variable_scope("word_emb"):
                self.inputs_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.inputs)
                # self.inputs_embedding_c = tf.nn.embedding_lookup(self.embedding_matrix_c, self.inputs)
            # with tf.variable_scope("cate_emb"):
            #     self.cate_embedding = tf.nn.embedding_lookup(self.aspect_embedding, tf.expand_dims(self.category, -1))
        #(self.batch_size, self.text_word_size) = tf.unstack(tf.shape(self.inputs))
        # (self.batch_size, self.target_word_size) = tf.unstack(tf.shape(self.targets))

    # def train(self, logits):
    #     with tf.variable_scope('train'):
    #         self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=logits)
    #         # self.cross_entropy = tf.nn.weighted_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
    #         # regu = tf.contrib.layers.l2_regularizer(0.00001, scope=None)
    #         # tvars = tf.trainable_variables()
    #         # self.loss_regu = tf.contrib.layers.apply_regularization(regu, tvars)
    #         # self.loss_cla = tf.reduce_mean(self.cross_entropy)
    #         # self.loss = self.loss_cla + self.loss_regu
    #
    #         self.loss = tf.reduce_mean(self.cross_entropy)
    #         # dif = tf.cast(self.labels, tf.float32) - self.logits_up
    #         # self.loss_up = tf.reduce_mean(dif * dif)
    #         # self.loss = self.loss_t + 0.1 * self.loss_up
    #
    #         self.accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, self.labels, 1), tf.float32))
    #
    #         tvars = tf.trainable_variables()
    #
    #         self.l2_loss = tf.contrib.layers.l2_regularizer(3.0)(tvars)#l2正则损失
    #
    #         # self.loss=self.loss+self.l2_loss
    #
    #         grads, global_norm = tf.clip_by_global_norm(
    #             tf.gradients(self.loss, tvars),
    #             self.max_grad_norm)
    #         tf.summary.scalar('global_grad_norm', global_norm)
    #
    #         opt = tf.train.AdamOptimizer(self.lr)
    #         # opt = tf.train.GradientDescentOptimizer(self.lr)
    #         # opt = tf.train.AdadeltaOptimizer(self.lr, rho=0.9, epsilon=1e-6)
    #
    #         self.train_op = opt.apply_gradients(
    #             zip(grads, tvars), name='train_op',
    #             global_step=self.global_step)
class CNN(Model):
    def __init__(
            self,config, embedding_matrix,sess,sequence_length,
             filter_sizes, num_filters):
        super(CNN, self).__init__(config, embedding_matrix)
        self.max_sequence_length=sequence_length
        self.position_embedding = tf.Variable(
            tf.random_uniform(shape=[self.max_sequence_length,50], minval=-1.0 * self.aspRandomBase,
                              maxval=self.aspRandomBase, seed=self.seed),
            name='position_embedding', dtype=tf.float32, trainable=True)
        # Placeholders for input, output and dropout
        num_classes=self.classes
        # self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")  # sequence_length为句子最大长度
        # self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

        # self.input_x = tf.placeholder(shape=(None, None), dtype=tf.int32, name='inputs')
        # self.labels = tf.placeholder(shape=(None,), dtype=tf.int32, name='labels')
        # self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/gpu:0'), tf.name_scope("embedding"):
            self.embedded_chars_expanded = tf.nn.embedding_lookup(self.embedding_matrix,
                                                         self.inputs)  # input_xshape=[None, sequence_length],return shape[None, sequence_length, embedding_size]
            self.posi_id_expanded= tf.nn.embedding_lookup(self.position_embedding,
                                                         self.positions)#(b*maxlen)*len*d
            embedding_size = self.embedding_size

            # #add position
            # self.embedded_chars_expanded=tf.concat([self.embedded_chars_expanded,self.posi_id_expanded],2)
            # embedding_size=self.embedding_size+50


            #self.cate_embedding_M = tf.nn.embedding_lookup(self.embedding_matrix, self.category)# b*L'*d
            # self.cate_embedding=tf.reduce_mean(self.cate_embedding_M,1, keep_dims=True)/self.inputs_len
            #self.cate_embedding = tf.reduce_mean(self.cate_embedding_M, 1, keep_dims=True)

        # W3 = tf.Variable(
        #     tf.random_uniform(shape=[self.aspnum, self.embedding_size], minval=-1.0 * self.aspRandomBase,
        #                       maxval=self.aspRandomBase, seed=self.seed),
        #     name='W3', dtype=tf.float32, trainable=True)
        V3 = tf.Variable(
            tf.random_uniform(shape=[self.embedding_size,1], minval=-1.0 * self.aspRandomBase,
                              maxval=self.aspRandomBase, seed=self.seed),
            name='V3', dtype=tf.float32, trainable=True)
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, num_filters]
                W1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W1",dtype=tf.float32, trainable=True)
                b1 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b1", dtype=tf.float32, trainable=True)
                # W2 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W2", dtype=tf.float32, trainable=True)
                # b2 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b2", dtype=tf.float32, trainable=True)
                conv1 = tf.nn.conv1d(
                    self.embedded_chars_expanded,
                    W1,
                    stride=1,
                    padding="VALID",
                    name="conv1")  # b * sequence_length-filter_size+1 *num_filters
                # conv2 = tf.nn.conv1d(
                #     self.embedded_chars_expanded,
                #     W2,
                #     stride=1,
                #     padding="VALID",
                #     name="conv2")  # b * sequence_length-filter_size+1 *num_filters
                # Apply nonlinearity
                h1=tf.nn.bias_add(conv1, b1, name="add")#b * sequence_length-filter_size+1 *1*num_filters
                # va = layers.fully_connected(self.cate_embedding, num_filters, weights_initializer=self.Winit,
                #                                        activation_fn=None)#b*1*num_filter
                # #va=tf.expand_dims(va, 1, name=None)
                # h2 = tf.nn.tanh(tf.nn.bias_add(conv2, b2)+va, name="tanh")#b * sequence_length-filter_size+1 **num_filters
                #
                # h=tf.multiply(h1,h2)
                h = tf.expand_dims(h1, -1, name=None)  # b*sequence_length-filter_size+1 *num_filters*1
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")  # b*1*num_filters*1
                pooled = tf.squeeze(pooled, 3)#b*1*num_filters
                pooled_outputs.append(pooled)#len(filter_size)*(b*1*num_filters)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 2)#b*1*(num_filters * len(filter_sizes)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])#b*(num_filters * len(filter_sizes)

        # Add dropout
        # with tf.name_scope("dropout"):
        #     self.h_drop = tf.nn.dropout(self.h_pool_flat,self.dropout_keep_proba)

        with tf.variable_scope('dropout'):
            self.h_pool_flat = layers.dropout(
                self.h_pool_flat, keep_prob=self.dropout_keep_proba,
                is_training=self.is_training,
            )

            # Final (unnormalized) scores and predictions

        with tf.name_scope("output"):
            # self.scores = layers.fully_connected(self.h_pool_flat, self.classes, weights_initializer=self.Winit,
            #                                      biases_initializer=self.Winit,
            #                                      activation_fn=None)  # b * len * 1

            W=tf.Variable(
                tf.random_uniform(shape=[num_filters_total, num_classes], minval=-1.0 * self.aspRandomBase,
                                  maxval=self.aspRandomBase, seed=self.seed),
                name='position_embedding', dtype=tf.float32, trainable=True)


            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer(seed=self.seed))
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_pool_flat, W, b, name="scores")


        with tf.variable_scope('classifier'):
            self.prediction = tf.argmax(self.scores,- 1, name="predictions")

        with tf.variable_scope('train'):
            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
                                                                                logits=self.scores)
            self.loss = tf.reduce_mean(self.cross_entropy)+3.0*l2_loss
            # self.loss = self.loss_senti

            self.accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.scores, self.labels, 1), tf.float32))

            tvars = tf.trainable_variables()
            grads, global_norm = tf.clip_by_global_norm(
                tf.gradients(self.loss, tvars),
                self.max_grad_norm)
            tf.summary.scalar('global_grad_norm', global_norm)

            #opt = tf.train.AdamOptimizer()
            #opt = tf.train.AdamOptimizer(self.lr)
            # opt = tf.train.GradientDescentOptimizer(self.lr)
            opt = tf.train.AdadeltaOptimizer(self.lr)
            # opt = tf.train.AdagradOptimizer(self.lr)

            self.train_op = opt.apply_gradients(
                zip(grads, tvars), name='train_op',
                global_step=self.global_step)

    def get_feed_data(self, x, y,p, e=None, class_weights=None, is_training=True):
        # x_m, x_sizes, xwordm = utils.batch(x)
        fd = {
            self.inputs: x,
            self.positions:p
        }
        if y is not None:
            fd[self.labels] = y
        if e is not None:
            fd[self.embedding_matrix] = e
        fd[self.is_training] = is_training
        return fd
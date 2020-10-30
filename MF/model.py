"""
Created on Mar 26, 2020

Matrix Factorization

@author: Gengziyao
"""
from utils import *
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class MF_layer(Layer):
    def __init__(self, user_num, item_num, latent_dim, use_bias=False, user_reg=1e-4, item_reg=1e-4,
                 user_bias_reg=1e-4, item_bias_reg=1e-4):
        """
        MF Layer
        :param user_num: user length
        :param item_num: item length
        :param latent_dim: latent number
        :param use_bias: whether using bias or not
        :param user_reg: regularization of user
        :param item_reg: regularization of item
        :param user_bias_reg: regularization of user bias
        :param item_bias_reg: regularization of item bias
        """
        super(MF_layer, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.latent_dim = latent_dim
        self.use_bias = use_bias
        self.user_reg = user_reg
        self.item_reg = item_reg
        self.user_bias_reg = user_bias_reg
        self.item_bias_reg = item_bias_reg

    def build(self, input_shape):
        self.p = self.add_weight(name='user_latent_matrix',
                                 shape=(self.user_num, self.latent_dim),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.user_reg),
                                 trainable=True)
        self.q = self.add_weight(name='item_latent_matrix',
                                 shape=(self.item_num, self.latent_dim),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.item_reg),
                                 trainable=True)
        self.user_bias = self.add_weight(name='user_bias',
                                         shape=(self.user_num, 1),
                                         initializer=tf.random_normal_initializer(),
                                         regularizer=l2(self.user_bias_reg),
                                         trainable=self.use_bias)
        self.item_bias = self.add_weight(name='user_bias',
                                         shape=(self.item_num, 1),
                                         initializer=tf.random_normal_initializer(),
                                         regularizer=l2(self.item_bias_reg),
                                         trainable=self.use_bias)

    def call(self, inputs, **kwargs):
        user_id, item_id, avg_score = inputs
        # MF
        latent_user = tf.nn.embedding_lookup(params=self.p, ids=user_id)
        latent_item = tf.nn.embedding_lookup(params=self.q, ids=item_id)
        outputs = tf.reduce_sum(tf.multiply(latent_user, latent_item), axis=1, keepdims=True)
        # MF-bias
        user_bias = tf.nn.embedding_lookup(params=self.user_bias, ids=user_id)
        item_bias = tf.nn.embedding_lookup(params=self.item_bias, ids=item_id)
        bias = tf.reshape((avg_score + user_bias + item_bias), shape=(-1, 1))
        # use bias
        outputs = bias + outputs if self.use_bias else outputs
        return outputs

    def summary(self):
        user_id = tf.keras.Input(shape=(), dtype=tf.int32)
        item_id = tf.keras.Input(shape=(), dtype=tf.int32)
        avg_score = tf.keras.Input(shape=(), dtype=tf.float32)
        tf.keras.Model(inputs=[user_id, item_id, avg_score], outputs=self.call([user_id, item_id, avg_score])).summary()


class MF(tf.keras.Model):
    def __init__(self, feature_columns, use_bias=False, user_reg=1e-4, item_reg=1e-4,
                 user_bias_reg=1e-4, item_bias_reg=1e-4):
        """
        MF Model
        :param feature_columns: dense_feature_columns + sparse_feature_columns
        :param use_bias: whether using bias or not
        :param user_reg: regularization of user
        :param item_reg: regularization of item
        :param user_bias_reg: regularization of user bias
        :param item_bias_reg: regularization of item bias
        """
        super(MF, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        num_users, num_items = self.sparse_feature_columns[0]['feat_num'], \
                               self.sparse_feature_columns[1]['feat_num']
        latent_dim = self.sparse_feature_columns[0]['embed_dim']
        self.mf_layer = MF_layer(user_num=num_users,
                                 item_num=num_items,
                                 latent_dim=latent_dim,
                                 use_bias=use_bias,
                                 user_reg=user_reg,
                                 item_reg=item_reg,
                                 user_bias_reg=user_bias_reg,
                                 item_bias_reg=item_bias_reg)

    def call(self, inputs, **kwargs):
        dense_inputs, sparse_inputs = inputs
        user_id, item_id = sparse_inputs[:, 0], sparse_inputs[:, 1]
        avg_score = dense_inputs
        outputs = self.mf_layer([user_id, item_id, avg_score])
        return outputs

    def summary(self, **kwargs):
        dense_inputs = tf.keras.Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = tf.keras.Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        tf.keras.Model(inputs=[dense_inputs, sparse_inputs], outputs=self.call([dense_inputs, sparse_inputs])).summary()


def Build_MF_Model(num_users, num_items, embedding_size):
    user_id = tf.keras.Input(shape=(1,), dtype=tf.int32, name='User_ID')
    movie_id = tf.keras.Input(shape=(1,), dtype=tf.int32, name='Movie_ID')
    user_avg_score = tf.keras.Input(shape=(1,), dtype=tf.float32, name='User_Avg_Score')
    user_embedding = tf.keras.layers.Embedding(input_dim=num_users,
                                               output_dim=embedding_size,
                                               input_length=1,
                                               embeddings_initializer="he_normal",
                                               embeddings_regularizer=tf.keras.regularizers.l2(1e-6),
                                               name="User_Embedding")(user_id)
    user_vectors = tf.keras.layers.Flatten(name='User_Vector')(user_embedding)
    user_bias = tf.keras.layers.Embedding(input_dim=num_users,
                                          output_dim=1,
                                          input_length=1,
                                          name="User_Bias_Embedding")(user_id)
    user_bias = tf.keras.layers.Flatten(name='User_Bias')(user_bias)
    movie_embedding = tf.keras.layers.Embedding(input_dim=num_items,
                                                output_dim=embedding_size,
                                                input_length=1,
                                                embeddings_initializer="he_normal",
                                                embeddings_regularizer=tf.keras.regularizers.l2(1e-6),
                                                name="Item_Embedding")(movie_id)
    movie_vectors = tf.keras.layers.Flatten(name='Item_Vector')(movie_embedding)
    movie_bias = tf.keras.layers.Embedding(input_dim=num_items,
                                           output_dim=1,
                                           input_length=1,
                                           name="Item_Bias_Embedding")(movie_id)
    movie_bias = tf.keras.layers.Flatten(name='Movie_Bias')(movie_bias)
    user_item_similar = tf.keras.layers.Dot(name="User_Item_Similar", axes=1)([user_vectors, movie_vectors])

    # avg_score = tf.keras.layers.Reshape((-1, 1))(user_avg_score)
    output = tf.keras.layers.Add(name="Score")([user_item_similar, user_bias, movie_bias, user_avg_score])
    # output = tf.keras.layers.Activation('sigmoid', name="Score")(output)

    return tf.keras.Model(inputs=[user_id, movie_id, user_avg_score], outputs=[output], name="MF")

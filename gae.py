import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU

import torch
from torch import nn
import torch.nn.functional as F

import os
from os.path import join


from dir_utils import create_dir
from tf_utils import is_cuda_available, print_summary


class GAE_torch(nn.Module):
    _logger = logging.getLogger(__name__)

    def __init__(self, d, hidden_size=10, l1_graph_penalty=0.01):
        super(GAE_torch, self).__init__()
        self.d = d #number of nodes in the graph
        self.hidden_size = hidden_size
        self.l1_graph_penalty = l1_graph_penalty


        self.encoder = nn.Sequential(nn.Linear(self.d, self.hidden_size), 
                                    nn.LeakyReLU(),
                                    nn.Linear(self.hidden_size, self.hidden_size),
                                    nn.LeakyReLU(),
                                    nn.Linear(self.hidden_size, self.d),
                                    nn.Sigmoid()) 
        self.adj_A = nn.Parameter(-0.1 + torch.randn(self.d,self.d)*(1/(self.d)))
        ind = np.diag_indices(self.adj_A.data.shape[0])
        self.adj_A.data[ind[0], ind[1]] = torch.zeros(self.adj_A.shape[0])
        self.decoder = nn.Sequential(nn.Linear(self.d, self.hidden_size), 
                                    nn.LeakyReLU(),
                                    nn.Linear(self.hidden_size, self.hidden_size),
                                    nn.LeakyReLU(),
                                    nn.Linear(self.hidden_size, self.d),
                                    nn.Sigmoid()) 
        
        self.useCUDA = torch.cuda.is_available()
    
    def _encode(self,x):
        out = self.encoder(x)
        return out

    def _decode(self,x):
        out = self.decoder(x)
        return out
    
    def forward(self, x):
        x = self._encode(x)
        x = x.reshape(x.shape[0],self.d,-1)
        x = torch.einsum('ijk,jl->ilk', x, self.adj_A)
        x = x.reshape(x.shape[0],-1)
        x = self._decode(x)
        x = x.reshape(x.shape[0], self.d)
        return x
    
    def loss(self,x ,x_hat, rho, alpha):
        exp_A = torch.matrix_power(torch.eye(self.d).double()+ torch.div(torch.matmul(self.adj_A.data,self.adj_A.data), self.d), self.d)
        h = torch.trace(exp_A) - self.d
        total_loss = (0.5 / x.shape[0]) * F.mse_loss(x,x_hat) \
                    + self.l1_graph_penalty * torch.norm(self.adj_A.data, 1) \
                    + alpha * h + 0.5 * rho * h * h
        return F.mse_loss(x, x_hat), h, total_loss

    def save_params(self, exDir):
        print('saving params...')
        torch.save(self.state_dict(), join(exDir, 'gae_params'))

    def load_params(self, exDir):
        print('loading params...')
        self.load_state_dict(torch.load(join(exDir, 'gae_params')))

class GAE(object):
    _logger = logging.getLogger(__name__)

    def __init__(self, n, d, x_dim, seed=8, num_encoder_layers=1, num_decoder_layers=1, hidden_size=5,
                 latent_dim=1, l1_graph_penalty=0, use_float64=False):
        self.n = n #number of samples in the dataset
        self.d = d #number of nodes in the graph
        self.x_dim = x_dim #2 for grayscale image, 3 for rbg etc.
        self.seed = seed
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.l1_graph_penalty = l1_graph_penalty
        self.tf_float_type = tf.dtypes.float64 if use_float64 else tf.dtypes.float32        

        # Initializer (for reproducibility)
        self.initializer = tf.keras.initializers.glorot_uniform(seed=self.seed)

        self._build()
        self._init_session()
        self._init_saver()

    def _init_session(self):
        if is_cuda_available():
            # Use GPU
            self.sess = tf.Session(config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    per_process_gpu_memory_fraction=0.5,
                    allow_growth=True,
                )
            ))
        else:
            self.sess = tf.Session()

    def _init_saver(self):
        self.saver = tf.train.Saver()

    def _build(self):
        tf.reset_default_graph()

        # Placeholders
        self.rho = tf.placeholder(self.tf_float_type)
        self.alpha = tf.placeholder(self.tf_float_type)
        self.lr = tf.placeholder(self.tf_float_type)
        self.X = tf.placeholder(self.tf_float_type, shape=[self.n, self.d, self.x_dim])

        # Variable for estimating graph
        W = tf.Variable(tf.random.uniform([self.d, self.d], minval=-0.1,maxval=0.1,
                                          dtype=self.tf_float_type, seed=self.seed))
        self.W_prime = self._preprocess_graph(W)

        # Losses
        self.mse_loss = self._get_mse_loss(self.X, self.W_prime)
        self.h = tf.linalg.trace(tf.linalg.expm(self.W_prime * self.W_prime)) - self.d    # Acyclicity
        self.loss = 0.5 / self.n * self.mse_loss \
                    + self.l1_graph_penalty * tf.norm(self.W_prime, ord=1) \
                    + self.alpha * self.h + 0.5 * self.rho * self.h * self.h

        # Train
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        self._logger.debug('Finished building Tensorflow graph')

    def _preprocess_graph(self, W):
        # Mask the diagonal entries of graph
        return tf.matrix_set_diag(W, tf.zeros(W.shape[0], dtype=self.tf_float_type))

    def _get_mse_loss(self, X, W_prime):
        X_prime = self._encoder_forward(X)
        X_prime = tf.einsum('ijk,jl->ilk', X_prime, W_prime)        
        X_prime = self._decoder_forward(X_prime)

        return tf.square(tf.linalg.norm(X - X_prime))

    def _encoder_forward(self, x):
        return self._flatten_forward(x, self.num_encoder_layers, self.hidden_size,
                                     input_dim=self.x_dim,
                                     output_dim=self.latent_dim)

    def _decoder_forward(self, x):
        return self._flatten_forward(x, self.num_decoder_layers, self.hidden_size,
                                     input_dim=self.latent_dim,
                                     output_dim=self.x_dim)

    def _flatten_forward(self, x, num_hidden_layers, hidden_size, input_dim, output_dim):
        if num_hidden_layers == 0:
            return x
        else:
            flatten = tf.reshape(x, shape=(-1, input_dim))
            for _ in range(num_hidden_layers):    # Hidden layer
                flatten = Dense(hidden_size, activation=None, kernel_initializer=self.initializer)(flatten)
                flatten = LeakyReLU(alpha=0.05)(flatten)

            flatten = Dense(output_dim, kernel_initializer=self.initializer)(flatten)    # Final output layer

            return tf.reshape(flatten, shape=(self.n, self.d, output_dim))

    def save(self, model_dir):
        create_dir(model_dir)
        self.saver.save(self.sess, '{}/model'.format(model_dir))

    @property
    def logger(self):
        try:
            return self._logger
        except:
            raise NotImplementedError('self._logger does not exist!')

if __name__ == '__main__':
    n, d = 3000, 20

    model = GAE(n, d, x_dim=1)
    model.print_summary(print)

    print('\nVariables needed by trainer:')
    print('model.sess: {}'.format(model.sess))
    print('model.train_op: {}'.format(model.train_op))
    print('model.loss: {}'.format(model.loss))
    print('model.mse_loss: {}'.format(model.mse_loss))
    print('model.h: {}'.format(model.h))
    print('model.W_prime: {}'.format(model.W_prime))
    print('model.X: {}'.format(model.X))
    print('model.rho: {}'.format(model.rho))
    print('model.alpha: {}'.format(model.alpha))
    print('model.lr: {}'.format(model.lr))

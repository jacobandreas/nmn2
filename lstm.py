#!/usr/bin/env python2

from util import Index

from data.util import pp
import logging
from lasagne import init, layers, objectives, updates
import numpy as np
import theano
import theano.tensor as T
import warnings

class LSTM:
  def __init__(self, params):
    batch_size = params["batch_size"]
    world_size = params["world_size"]
    vocab_size = params["vocab_size"]
    hidden_size = params["hidden_size"]
    output_size = params["output_size"]

    self.batch_size = batch_size

    self.l_input_text = layers.InputLayer((batch_size, None))
    self.l_input_world = layers.InputLayer((batch_size, world_size))

    self.l_embed_text = layers.EmbeddingLayer(self.l_input_text, vocab_size, hidden_size)

    self.l_dense_world_1 = layers.DenseLayer(self.l_input_world, hidden_size)
    self.l_dense_world_2 = layers.DenseLayer(self.l_dense_world_1, hidden_size)

    self.l_reshape_world = layers.ReshapeLayer(self.l_dense_world_2, (batch_size, 1, hidden_size))

    self.l_concat = layers.ConcatLayer([self.l_reshape_world, self.l_embed_text])

    self.l_forward_1 = layers.LSTMLayer(self.l_concat, hidden_size)
    self.l_forward_2 = layers.LSTMLayer(self.l_forward_1, hidden_size)
    self.l_slice = layers.SliceLayer(self.l_forward_2, indices=-1, axis=1)

    self.l_predict = layers.DenseLayer(self.l_slice, output_size, nonlinearity=None)

    self.t_input_text = T.imatrix("input_text")
    self.t_input_world = T.matrix("input_world")
    self.input_mapping = {self.l_input_text: self.t_input_text,
                          self.l_input_world: self.t_input_world}
    self.t_output = layers.get_output(self.l_predict, self.input_mapping)
    self.t_target = T.matrix("target")
    self.t_loss = T.mean(objectives.squared_error(self.t_output, self.t_target))
    self.params = layers.get_all_params(self.l_predict)

    momentum = 0.9
    lr = .00001
    upd = updates.momentum(self.t_loss, self.params, lr, momentum)

    loss_inputs = [self.t_input_text, self.t_input_world, self.t_target]
    self.f_loss = theano.function(loss_inputs, self.t_loss)
    self.f_train = theano.function(loss_inputs, self.t_loss, updates=upd)

    self.index = Index()

  def linearize(self, query):
    pretty = pp(query)
    sep = pretty.replace("(", "( ").replace(")", " )")
    indices = [self.index.index(t) for t in sep.split()]
    return np.tile(indices, (self.batch_size, 1)).astype(np.int32)

  def train(self, query, world, output):
    lq = self.linearize(query)
    #print lq.shape
    #print world.shape
    #print output.shape
    return self.f_train(lq, world, output)

  def loss(self, query, world, output):
    lq = self.linearize(query)
    return self.f_loss(lq, world, output)

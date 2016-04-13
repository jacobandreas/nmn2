#!/usr/bin/env python2

import lasagne
import numpy as np
import theano
import theano.tensor as T

class Regression:
  def __init__(self):
    self.dtype = theano.config.floatX
    self.loss = lasagne.objectives.squared_error

  def make_target(self):
    return T.matrix("target")

class Classification:
  def __init__(self):
    self.dtype = np.int32
    self.loss = lasagne.objectives.categorical_crossentropy

  def make_target(self):
    return T.ivector("target")

class ImageOutput:
  def __init__(self):
    self.dtype = theano.config.floatX
    self.loss = lasagne.objectives.squared_error

  def make_target(self):
    return T.tensor4("target")

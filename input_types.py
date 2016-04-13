#!/usr/bin/env python2

import numpy as np
import theano
import theano.tensor as T

class VectorInput:
  def __init__(self):
    self.dtype = theano.config.floatX
  
  def make_input(self):
    return T.matrix("input")


class ImageInput:
  def __init__(self):
    self.dtype = theano.config.floatX

  def make_input(self):
    return T.tensor4("input")

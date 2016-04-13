#!/usr/bin/env python2

import nmn
from lasagne import nonlinearities
from debug import HackL1ConvModule, HackL2ConvModule, HackL3Module, HackColorModule

class ShapeModuleBuilder:
  def __init__(self, params):
    self.batch_size = params["batch_size"]
    self.image_size = params["image_size"]
    self.channels = params["channels"]
    self.mid_filters = params["mid_filters"]
    self.out_filters = params["out_filters"]
    self.input_mid_filter_size = params["input_mid_filter_size"]
    self.input_out_filter_size = params["input_out_filter_size"]
    self.filter_size = params["filter_size"]
    self.input_mid_pool = params["input_mid_pool"]
    self.input_out_pool = params["input_out_pool"]

    self.hidden_size = params["hidden_size"]
    self.vocab_size = params["vocab_size"]

    self.message_size = self.image_size / (self.input_mid_pool *
        self.input_out_pool)

  def build(self, name, arity):
    if name == "_output":
      #return nmn.IdentityModule()
      return nmn.MLPModule(self.batch_size, self.out_filters * self.message_size *
          self.message_size, self.hidden_size, self.vocab_size,
          output_nonlinearity=nonlinearities.softmax)

    elif arity == 0:
      #return nmn.Conv1Module(self.batch_size, self.channels, self.image_size,
      #    self.out_filters, self.input_mid_filter_size,
      #    self.image_size / self.message_size)

      return nmn.ConvModule(self.batch_size, self.channels, self.image_size,
          self.mid_filters, self.out_filters, self.input_mid_filter_size,
          self.input_out_filter_size, self.input_mid_pool, self.input_out_pool)

      #return nmn.MLPModule(self.batch_size, self.channels * self.image_size *
      #    self.image_size, self.hidden_size, self.out_filters *
      #    self.message_size * self.message_size)

      #channel = {"red": 0, "green": 1, "blue": 2}[name]
      #return HackColorModule(channel)

    elif arity == 1:
      #r, c = {
      #    "left_of": (1, 0),
      #    "above": (0, 1)
      #}[name]
      #return HackL2ConvModule(r, c)

      #return nmn.ConvModule(self.batch_size, self.out_filters, self.message_size,
      #    self.mid_filters, self.out_filters, self.filter_size,
      #    self.filter_size, 1, 1, tie=True)

      return nmn.MLPModule(self.batch_size, self.out_filters * self.message_size
          * self.message_size, self.hidden_size, self.out_filters * self.message_size
          * self.message_size)

    elif arity == 2:
      return HackL3Module()

      #return nmn.MLPModule(self.batch_size, 2 * self.out_filters * self.message_size
      #    * self.message_size, self.hidden_size, self.out_filters * self.message_size
      #    * self.message_size)

      #return nmn.ConvModule(self.batch_size, self.out_filters * 2,
      #    self.message_size, self.mid_filters, self.out_filters,
      #    self.filter_size, 1, 1)

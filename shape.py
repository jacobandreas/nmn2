#!/usr/bin/env python2

import nmn
from lasagne import nonlinearities

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
            return nmn.MLPModule(self.batch_size, self.out_filters * self.message_size *
                    self.message_size, self.hidden_size, self.vocab_size,
                    output_nonlinearity=nonlinearities.softmax)

        elif arity == 0:
            return nmn.ConvModule(self.batch_size, self.channels, self.image_size,
                    self.mid_filters, self.out_filters, self.input_mid_filter_size,
                    self.input_out_filter_size, self.input_mid_pool, self.input_out_pool)

        elif arity == 1:
            return nmn.MLPModule(self.batch_size, self.out_filters * self.message_size
                    * self.message_size, self.hidden_size, self.out_filters * self.message_size
                    * self.message_size)

        elif arity == 2:
            return nmn.CombineModule()

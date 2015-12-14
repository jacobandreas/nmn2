#!/usr/bin/env python2

from misc.indices import QUESTION_INDEX, ANSWER_INDEX, UNK_ID
from opt import adadelta

import apollocaffe
from apollocaffe.layers import *
import numpy as np

class StoredProgramModel:
    def __init__(self, config, opt_config):
        self.config = config
        self.opt_config = opt_config
        self.opt_state = adadelta.State()
        self.apollo_net = apollocaffe.ApolloNet()

    def train(self):
        self.apollo_net.backward()
        adadelta.update(self.apollo_net, self.opt_state, self.opt_config)
        #self.apollo_net.update(
        #        lr=0.1, momentum=0.9, clip_gradients=10.0)

    def reset(self):
        self.loss_counter = 0
        self.att_counter = 0
        self.apollo_net.clear_forward()

    @profile
    def forward(self, layout_type, batch_layout_labels, question_data,
            image_data, dropout):
        images = self.forward_image_data(image_data, dropout)
        question_hidden = self.forward_lstm(question_data, dropout)
        att1_hidden = self.forward_att(question_hidden, images)
        #att2_hidden = self.forward_att(att1_hidden)
        self.pred_layer = self.forward_pred(att1_hidden)

        self.prediction_data = self.apollo_net.blobs[self.pred_layer].data
        self.att_data = self.apollo_net.blobs["att_softmax_0"].data.reshape((-1, 14, 14))
        #self.att_data = np.zeros((100, 14, 14))

    @profile
    def forward_image_data(self, image_data, dropout):

        self.batch_size = image_data.shape[0]
        self.channels = image_data.shape[1]
        self.image_size = image_data.shape[2] * image_data.shape[3]

        image_data_rs = image_data.reshape(
                (self.batch_size, self.channels, self.image_size, 1))

        net = self.apollo_net

        images = "data_images"
        images_dropout = "data_images_dropout"

        if images not in net.blobs:
            net.f(DummyData(images, image_data_rs.shape))
        net.blobs[images].data[...] = image_data_rs

        if dropout:
            net.f(Dropout(images_dropout, 0.5, bottoms=[images]))
            return images_dropout
        else:
            return images

    @profile
    def forward_lstm(self, question_data, dropout):
        net = self.apollo_net
        batch_size, length = question_data.shape
        assert batch_size == self.batch_size

        wordvec_param = "wordvec_param"
        input_value_param = "input_value_param"
        input_gate_param = "input_gate_param"
        forget_gate_param = "forget_gate_param"
        output_gate_param = "output_gate_param"

        seed = "lstm_seed"
        final_hidden = "lstm_final_hidden"

        prev_hidden = seed
        prev_mem = seed

        net.f(NumpyData(seed, np.zeros((batch_size, self.config.lstm_hidden))))

        for t in range(length):
            word = "lstm_word_%d" % t
            wordvec = "lstm_wordvec_%d" % t
            concat = "lstm_concat_%d" % t
            lstm = "lstm_unit_%d" % t

            hidden = "lstm_hidden_%d" % t
            mem = "lstm_mem_%d" % t

            net.f(NumpyData(word, question_data[:,t]))

            net.f(Wordvec(
                    wordvec, self.config.lstm_hidden, len(QUESTION_INDEX),
                    bottoms=[word], param_names=[wordvec_param]))

            net.f(Concat(concat, bottoms=[prev_hidden, wordvec]))

            net.f(LstmUnit(
                    lstm, self.config.lstm_hidden, bottoms=[concat, prev_mem],
                    param_names=[input_value_param, input_gate_param,
                                forget_gate_param, output_gate_param],
                    tops=[hidden, mem]))

            prev_hidden = hidden
            prev_mem = mem

        if dropout:
            net.f(Dropout("lstm_dropout", 0.5, bottoms=[prev_hidden]))
            net.f(InnerProduct(
                    final_hidden, self.config.att_hidden, bottoms=["lstm_dropout"]))

        else:
            net.f(InnerProduct(
                    final_hidden, self.config.att_hidden, bottoms=[prev_hidden]))

        return final_hidden

    @profile
    def forward_att(self, last_hidden, images):
        net = self.apollo_net

        proj_image = "att_proj_image_%d" % self.att_counter
        tile_hidden = "att_broadcast_hidden_%d" % self.att_counter
        add = "att_sum_%d" % self.att_counter
        relu = "att_relu_%d" % self.att_counter
        mask = "att_mask_%d" % self.att_counter
        softmax = "att_softmax_%d" % self.att_counter

        tile_mask = "att_tile_mask_%d" % self.att_counter
        weight = "att_weight_%d" % self.att_counter
        reduction = "att_reduction_%d" % self.att_counter
        ip = "att_ip_%d" % self.att_counter

        comb = "att_comb_%d" % self.att_counter

        # compute attention mask

        net.f(Convolution(
                proj_image, (1,1), self.config.att_hidden,
                bottoms=[images]))

        net.blobs[last_hidden].reshape(
                (self.batch_size, self.config.att_hidden, 1, 1))

        net.f(Tile(
                tile_hidden, axis=2, tiles=self.image_size,
                bottoms=[last_hidden]))

        net.f(Eltwise(add, "SUM", bottoms=[proj_image, tile_hidden]))

        net.f(ReLU(relu, bottoms=[add]))

        net.f(Convolution(mask, (1, 1), 1, bottoms=[relu]))

        net.blobs[mask].reshape((self.batch_size, self.image_size))

        net.f(Softmax(softmax, bottoms=[mask]))

        # TODO WTF
        net.f(Power("copy_softmax", bottoms=[softmax]))
        net.blobs["copy_softmax"].reshape((self.batch_size, 1, self.image_size, 1))

        # compute average features

        net.f(Tile(tile_mask, axis=1, tiles=self.channels, bottoms=["copy_softmax"]))

        net.f(Eltwise(weight, "PROD", bottoms=[tile_mask, images]))

        # reduction
        net.f(InnerProduct(
            reduction, 1, axis=2, bottoms=[weight],
            weight_filler=Filler("constant", 1),
            bias_filler=Filler("constant", 0), param_lr_mults=[0, 0]))

        net.f(InnerProduct(
                ip, self.config.att_hidden, bottoms=[reduction]))
                #weight_filler=Filler("uniform", 0.01)))
                #param_lr_mults=[0.1, 0.1]))

        net.blobs[ip].reshape(
                (self.batch_size, self.config.att_hidden, 1, 1))

        net.f(Power(
            "scale", scale=0.1, bottoms=[ip]))

        #print np.squeeze(net.blobs[ip].data[0,...])
        #print np.linalg.norm(net.blobs[ip].data)
        #print np.linalg.norm(net.blobs[last_hidden].data)
        #print

        # combine with previous hidden

        net.f(Eltwise(comb, "SUM", bottoms=["scale", last_hidden]))
        #net.f(Eltwise(comb, "SUM", bottoms=[ip, last_hidden]))

        return comb

    def forward_pred(self, last_hidden):
        net = self.apollo_net

        hidden_relu = "answer_hidden_relu"
        ip = "answer_ip"

        net.f(ReLU(hidden_relu, bottoms=[last_hidden]))
        net.f(InnerProduct(ip, len(ANSWER_INDEX), bottoms=[hidden_relu]))

        return ip

    def loss(self, answers):
        net = self.apollo_net

        loss_data = "loss_data_%d" % self.loss_counter
        loss_score = "loss_score_%d" % self.loss_counter

        #print net.blobs[self.pred_layer].data[:10,:10]
        #print answers[:10]
        #exit()

        net.f(NumpyData(loss_data, answers))

        self.loss_counter += 1

        loss = net.f(SoftmaxWithLoss(
                loss_score, bottoms=[self.pred_layer, loss_data],
                ignore_label=UNK_ID))

        return loss

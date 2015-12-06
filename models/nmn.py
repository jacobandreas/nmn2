from layers.reinforce import Index, AsLoss
from misc.indices import QUESTION_INDEX, MODULE_INDEX, ANSWER_INDEX, UNK_ID
from misc import util
from opt import adadelta

import apollocaffe
from apollocaffe.layers import *
import numpy as np

class Module:
    def __init__(self, config):
        self.config = config

        if hasattr(config, "pred_hidden"):
            self.pred_size = config.pred_hidden
        else:
            self.pred_size = len(ANSWER_INDEX)

    def forward(self, index, label_data, bottoms, image, dropout, apollo_net):
        raise NotImplementedError()

    def __str__(self):
        return self.__class__.__name__

class AttendModule(Module):
    def forward(self, index, label_data, bottoms, image, dropout, apollo_net):
        assert len(bottoms) == 0

        net = apollo_net

        batch_size, channels, image_size, trailing = net.blobs[image].shape
        assert trailing == 1

        proj_image = "Attend_%d_proj_image" % index
        label = "Attend_%d_label" % index
        label_vec = "Attend_%d_label_vec" % index
        label_vec_dropout = "Attend_%d_label_vec_dropout" % index
        tile = "Attend_%d_tile" % index
        sum = "Attend_%d_sum" % index
        relu = "Attend_%d_relu" % index
        mask = "Attend_%d_mask" % index
        softmax = "Attend_%d_softmax" % index
        copy_softmax = "Attend_%d_copy_softmax" % index

        proj_image_param_weight = "Attend_proj_image_param_weight"
        proj_image_param_bias = "Attend_proj_image_param_bias"
        label_vec_param = "Attend_label_vec_param"
        mask_param_weight = "Attend_mask_param_weight"
        mask_param_bias = "Attend_mask_param_bias"

        # compute attention mask

        net.f(Convolution(
                proj_image, (1, 1), self.config.att_hidden, bottoms=[image],
                param_names=[proj_image_param_weight, proj_image_param_bias]))
        net.f(NumpyData(label, label_data))
        net.f(Wordvec(
                label_vec, self.config.att_hidden, len(MODULE_INDEX),
                bottoms=[label], param_names=[label_vec_param]))
        net.blobs[label_vec].reshape((batch_size, self.config.att_hidden, 1, 1))
        #if dropout:
        #    net.f(Dropout(label_vec_dropout, 0.5, bottoms=[label_vec]))
        #    label_vec_final = label_vec_dropout
        #else:
        #    label_vec_final = label_vec
        label_vec_final = label_vec
        net.f(Tile(tile, axis=2, tiles=image_size, bottoms=[label_vec_final]))
        net.f(Eltwise(sum, "SUM", bottoms=[proj_image, tile]))
        net.f(ReLU(relu, bottoms=[sum]))
        net.f(Convolution(mask, (1, 1), 1, bottoms=[relu],
                param_names=[mask_param_weight, mask_param_bias]))
        net.blobs[mask].reshape((batch_size, image_size))
        net.f(Softmax(softmax, bottoms=[mask]))
        # TODO still WTF
        net.f(Power(copy_softmax, bottoms=[softmax]))
        net.blobs[copy_softmax].reshape((batch_size, 1, image_size, 1))

        return copy_softmax

#class ReAttendModule(Module):
#    pass
#
#class CombineModule(Module):
#    pass

class ClassifyModule(Module):
    def forward(self, index, label_data, bottoms, image, dropout, apollo_net):
        assert len(bottoms) == 1
        mask = bottoms[0]

        net = apollo_net

        batch_size, channels, image_size, _ = net.blobs[image].shape

        tile_mask = "Classify_%d_tile_mask" % index
        weight = "Classify_%d_weight" % index
        reduction = "Classify_%d_reduction" % index
        ip = "Classify_%d_ip" % index
        scale = "Classify_%d_scale" % index

        reduction_param_weight = "Classify_reduction_param_weight"
        reduction_param_bias = "Classify_reduction_param_bias"
        ip_param_weight = "Classify_ip_param_weight"
        ip_param_bias = "Classify_ip_param_bias"

        net.f(Tile(tile_mask, axis=1, tiles=channels, bottoms=[mask]))
        net.f(Eltwise(weight, "PROD", bottoms=[tile_mask, image]))
        net.f(InnerProduct(
            reduction, 1, axis=2, bottoms=[weight],
            weight_filler=Filler("constant", 1),
            bias_filler=Filler("constant", 0),
            param_lr_mults=[0, 0],
            param_names=[reduction_param_weight, reduction_param_bias]))
        net.f(InnerProduct(
            ip, self.pred_size, bottoms=[reduction],
            param_names=[ip_param_weight, ip_param_bias]))
        net.f(Power(scale, scale=0.1, bottoms=[ip]))

        return scale


class MeasureModule(Module):
    def forward(self, index, label_data, bottoms, image, dropout, apollo_net):
        assert len(bottoms) == 1
        mask = bottoms[0]

        net = apollo_net

        ip = "Measure_%d_ip"
        ip_param_weight = "Measure_ip_param_weight"
        ip_param_bias = "Measure_ip_param_bias"

        net.f(InnerProduct(
            ip, self.pred_size, bottoms=[mask],
            param_names=[ip_param_weight, ip_param_bias]))

        return ip


class Nmn:
    def __init__(self, modules, apollo_net):
        self.apollo_net = apollo_net

        # TODO eww
        counter = [0]
        def number(tree):
            r = counter[0]
            counter[0] += 1
            return r
        numbered = util.tree_map(number, modules)
        assert util.flatten(numbered) == range(len(numbered))

        def children(tree):
            if not isinstance(tree, tuple):
                return ()
            return tuple(c[0] if isinstance(c, tuple) else c for c in tree[1:])
        child_annotated = util.tree_map(children, numbered)

        self.modules = util.flatten(modules)
        self.children = child_annotated

    def forward(self, label_data, image, dropout):
        flat_data = util.flatten(label_data)
        outputs = [None for i in range(len(self.modules))]
        for i in reversed(range(len(self.modules))):
            bottoms = [outputs[j] for j in self.children[i]]
            assert None not in bottoms
            output = self.modules[i].forward(
                    i, label_data[i], bottoms, image, dropout, self.apollo_net)
            outputs[i] = output

        return outputs[0]

class NmnModel:
    def __init__(self, config, opt_config):
        self.config = config
        self.opt_config = opt_config
        self.opt_state = adadelta.State()

        self.nmns = dict()

        self.apollo_net = apollocaffe.ApolloNet()

    def get_nmn(self, modules):
        if modules not in self.nmns:
            self.nmns[modules] = Nmn(modules, self.apollo_net)
        return self.nmns[modules]


    def forward(self, layouts, layout_data, question_data, image_data, dropout):

        # predict layout

        question_hidden = self.forward_question(question_data, dropout)
        layout_ids, layout_probs = self.forward_layout(question_hidden, layout_data)
        chosen_layouts = [ll[i] for ll, i in zip(layouts, layout_ids)]
        assert len(set(l.modules for l in chosen_layouts)) == 1
        modules = chosen_layouts[0].modules
        layout_label_data = np.asarray([l.labels for l in chosen_layouts])

        self.layout_ids = layout_ids
        self.layout_probs = layout_probs

        # predict answer

        nmn = self.get_nmn(modules)

        image = self.forward_image(image_data, dropout)
        nmn_hidden = nmn.forward(layout_label_data, image, dropout)
        self.prediction = self.forward_pred(question_hidden, nmn_hidden)

        self.prediction_data = self.apollo_net.blobs[self.prediction].data
        self.att_data = self.apollo_net.blobs["Attend_1_softmax"].data
        self.att_data = self.att_data.reshape((-1, 14, 14))

    def forward_layout(self, question_hidden, layout_data):
        net = self.apollo_net
        batch_size, n_layouts = layout_data.shape

        ip = "LAYOUT_ip"
        softmax = "LAYOUT_softmax"

        net.f(InnerProduct(ip, layout_data.shape[1], bottoms=[question_hidden]))
        net.f(Softmax(softmax, bottoms=[ip]))

        return [0 for i in range(batch_size)], softmax

    def forward_image(self, image_data, dropout):

        batch_size = image_data.shape[0]
        channels = image_data.shape[1]
        image_size = image_data.shape[2] * image_data.shape[3]

        image_data_rs = image_data.reshape(
                (batch_size, channels, image_size, 1))

        net = self.apollo_net

        images = "IMAGE_data"
        images_dropout = "IMAGE_dropout"

        if images not in net.blobs:
            net.f(DummyData(images, image_data_rs.shape))
        net.blobs[images].data[...] = image_data_rs

        if dropout:
            net.f(Dropout(images_dropout, 0.5, bottoms=[images]))
            return images_dropout
        else:
            return images

    def forward_question(self, question_data, dropout):
        net = self.apollo_net
        batch_size, length = question_data.shape

        wordvec_param = "QUESTION_wordvec_param"
        input_value_param = "QUESTION_input_value_param"
        input_gate_param = "QUESTION_input_gate_param"
        forget_gate_param = "QUESTION_forget_gate_param"
        output_gate_param = "QUESTION_output_gate_param"

        seed = "QUESTION_lstm_seed"
        dropout = "QUESTION_lstm_dropout"
        final_hidden = "QUESTION_lstm_final_hidden"

        prev_hidden = seed
        prev_mem = seed

        net.f(NumpyData(seed, np.zeros((batch_size, self.config.lstm_hidden))))

        for t in range(length):
            word = "QUESTION_lstm_word_%d" % t
            wordvec = "QUESTION_lstm_wordvec_%d" % t
            concat = "QUESTION_lstm_concat_%d" % t
            lstm = "QUESTION_lstm_unit_%d" % t

            hidden = "QUESTION_lstm_hidden_%d" % t
            mem = "QUESTION_lstm_mem_%d" % t

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

        # TODO consolidate with module?
        if hasattr(self.config, "pred_hidden"):
            pred_size = self.config.pred_hidden
        else:
            pred_size = len(ANSWER_INDEX)

        if dropout:
            net.f(Dropout(dropout, 0.5, bottoms=[prev_hidden]))
            net.f(InnerProduct(final_hidden, pred_size, bottoms=[dropout]))
        else:
            net.f(InnerProduct(final_hidden, pred_size, bottoms=[prev_hidden]))

        return final_hidden

    def forward_pred(self, question_hidden, nmn_hidden):
        net = self.apollo_net

        sum = "PRED_sum"
        relu = "PRED_relu"
        ip = "PRED_ip"

        net.f(Eltwise(sum, "SUM", bottoms=[question_hidden, nmn_hidden]))

        if hasattr(self.config, "pred_hidden"):
            net.f(ReLU(relu, bottoms=[sum]))
            net.f(InnerProduct(ip, len(ANSWER_INDEX), bottoms=[relu]))
            return ip
        else:
            return sum

    def loss(self, answer_data):
        net = self.apollo_net

        target = "PRED_target_%d" % self.loss_counter
        loss = "PRED_loss_%d" % self.loss_counter
        softmax = "PRED_softmax_%d" % self.loss_counter
        self.loss_counter += 1

        net.f(NumpyData(target, answer_data))
        loss = net.f(SoftmaxWithLoss(
            loss, bottoms=[self.prediction, target], ignore_label=UNK_ID))

        net.f(Softmax(softmax, bottoms=[self.prediction]))

        pred_probs = net.blobs[softmax].data
        batch_size = pred_probs.shape[0]
        pred_ans_probs = pred_probs[np.arange(batch_size), answer_data.astype(np.int)]
        pred_ans_log_probs = np.log(pred_ans_probs)
        pred_ans_log_probs[answer_data == UNK_ID] = 0

        return loss, -pred_ans_log_probs

    def reinforce(self, losses):
        net = self.apollo_net

        choice_data = "REINFORCE_choice_data"
        loss_data = "REINFORCE_loss_data"
        index = "REINFORCE_index"
        weight = "REINFORCE_weight"
        reduction = "REINFORCE_reduction"
        loss = "REINFORCE_loss"

        net.f(NumpyData(choice_data, self.layout_ids))
        net.f(NumpyData(loss_data, losses))
        net.f(Index(index, {}, bottoms=[self.layout_probs, choice_data]))
        net.f(Eltwise(weight, "PROD", bottoms=[index, loss_data]))
        from layers.reinforce import Reduction
        net.f(Reduction(reduction, 0, bottoms=[weight], loss_weight=1.0))

    def reset(self):
        self.apollo_net.clear_forward()
        self.loss_counter = 0
        self.question_hidden = None

    def train(self):
        self.apollo_net.backward()
        adadelta.update(self.apollo_net, self.opt_state, self.opt_config)

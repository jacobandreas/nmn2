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
        const_label = "Attend_%d_const_label" % index
        label_vec = "Attend_%d_label_vec" % index
        const_label_vec = "Attend_%d_const_label_vec" % index
        label_vec_dropout = "Attend_%d_label_vec_dropout" % index
        tile = "Attend_%d_tile" % index
        sum = "Attend_%d_sum" % index
        sum_vec = "Attend_%d_sum_vec" % index
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

        #net.f(NumpyData(label, label_data))
        #net.f(NumpyData(const_label, np.ones(label_data.shape) * UNK_ID))
        #net.f(Wordvec(
        #        label_vec, self.config.att_hidden, len(MODULE_INDEX),
        #        bottoms=[label], param_names=[label_vec_param]))
        #net.f(Wordvec(
        #        const_label_vec, self.config.att_hidden, len(MODULE_INDEX),
        #        bottoms=[label], param_names=[label_vec_param]))
        #net.f(Eltwise(sum_vec, "SUM", bottoms=[label_vec, const_label_vec]))
        #net.blobs[sum_vec].reshape((batch_size, self.config.att_hidden, 1, 1))
        #net.f(Tile(tile, axis=2, tiles=image_size, bottoms=[sum_vec]))
        #net.f(Eltwise(sum, "PROD", bottoms=[tile, proj_image]))
        #net.f(Convolution(
        #    mask, (1, 1), 1, bottoms=[sum], 
        #    param_names=[mask_param_weight, mask_param_bias]))
        #    #param_lr_mults=[0,0],
        #    #weight_filler=Filler("constant", 1),
        #    #bias_filler=Filler("constant", 0)))
        #return mask

        net.f(NumpyData(label, label_data))
        net.f(Wordvec(
                label_vec, self.config.att_hidden, len(MODULE_INDEX),
                bottoms=[label], param_names=[label_vec_param]))
        net.blobs[label_vec].reshape((batch_size, self.config.att_hidden, 1, 1))
        if dropout:
            net.f(Dropout(label_vec_dropout, 0.5, bottoms=[label_vec]))
            label_vec_final = label_vec_dropout
        else:
            label_vec_final = label_vec

        #net.f(Power(label_vec, bottoms=[qh]))
        #net.blobs[label_vec].reshape((batch_size, self.config.att_hidden, 1, 1))
        #label_vec_final = label_vec

        net.f(Tile(tile, axis=2, tiles=image_size, bottoms=[label_vec_final]))
        
        net.f(Eltwise(sum, "SUM", bottoms=[proj_image, tile]))
        net.f(ReLU(relu, bottoms=[sum]))
        net.f(Convolution(mask, (1, 1), 1, bottoms=[relu],
                param_names=[mask_param_weight, mask_param_bias]))
                #param_lr_mults=[0.1, 0.1]))
        return mask


#class ReAttendModule(Module):
#    pass
#

#class CombineModule(Module):
#    def forward(self, index, label_data, bottoms, image, dropout, apollo_net, qh=None):
#        net = apollo_net
#        assert len(bottoms) >= 2
#        assert all(net.blobs[l].shape[1] == 1 for l in bottoms)
#
#        concat = "Combine_%d_concat" % index
#        conv = "Combine_%d_conv" % index
#        relu = "Combine_%d_relu" % index
#
#        net.f(Concat(concat, axis=1, bottoms=bottoms))
#        net.f(Convolution(conv, (1, 1), 1, bottoms=[concat]))
#        net.f(ReLU(relu, bottoms=[conv]))
#
#        return relu

class CombineModule(Module):
    def forward(self, index, label_data, bottoms, image, dropout, apollo_net):
        net = apollo_net
        assert len(bottoms) >= 2
        assert all(net.blobs[l].shape[1] == 1 for l in bottoms)
        batch_size, channels, image_size, _ = net.blobs[bottoms[0]].shape

        prod = "Combine_%d_prod" % index
        sum = "Combine_%d_sum" % index
        avg = "Combine_%d_avg" % index
        softmax = "Combine_%d_softmax" % index
        copy_softmax = "Combine_%d_copy_softmax" % index

        net.f(Eltwise(sum, "SUM", bottoms=bottoms))
        #net.f(Power(avg, scale=0.5, bottoms=[sum]))

        #for b in bottoms:
        #    print net.blobs[b].data[0,...,0]

        return sum

        ##dummy = "Combine_%d_dummy" % index
        ##net.f(NumpyData(dummy, np.zeros(net.blobs[bottoms[0]].shape)))
        ##return dummy

class StupidModule(Module):
    def forward(self, index, label_data, bottoms, image, dropout, apollo_net):
        assert len(bottoms) == 0

        net = apollo_net

        net.f(InnerProduct("Stupid_ip1", self.config.att_hidden, bottoms=[image]))
        net.f(ReLU("Stupid_relu1", bottoms=["Stupid_ip1"]))
        net.f(InnerProduct("Stupid_ip2", self.config.att_hidden, bottoms=["Stupid_relu1"]))

        return "Stupid_ip2"

class ClassifyModule(Module):
    def forward(self, index, label_data, bottoms, image, dropout, apollo_net):
        assert len(bottoms) == 1
        mask = bottoms[0]

        net = apollo_net

        batch_size, channels, image_size, _ = net.blobs[image].shape

        softmax = "Classify_%d_softmax" % index
        copy_softmax = "Classify_%d_copy_softmax" % index
        tile_mask = "Classify_%d_tile_mask" % index
        weight = "Classify_%d_weight" % index
        reduction = "Classify_%d_reduction" % index
        ip = "Classify_%d_ip" % index
        scale = "Classify_%d_scale" % index

        label = "Classify_%d_label" % index
        label_vec = "Classify_%d_label_vec" % index
        concat = "Classify_%d_concat" % index

        reduction_param_weight = "Classify_reduction_param_weight"
        reduction_param_bias = "Classify_reduction_param_bias"
        ip_param_weight = "Classify_ip_param_weight"
        ip_param_bias = "Classify_ip_param_bias"

        net.blobs[mask].reshape((batch_size, image_size))
        net.f(Softmax(softmax, bottoms=[mask]))
        # TODO still WTF
        net.f(Power(copy_softmax, bottoms=[softmax]))
        net.blobs[copy_softmax].reshape((batch_size, 1, image_size, 1))

        net.f(Tile(tile_mask, axis=1, tiles=channels, bottoms=[copy_softmax]))
        net.f(Eltwise(weight, "PROD", bottoms=[tile_mask, image]))
        net.f(InnerProduct(
                reduction, 1, axis=2, bottoms=[weight],
                weight_filler=Filler("constant", 1),
                bias_filler=Filler("constant", 0),
                param_lr_mults=[0, 0],
                param_names=[reduction_param_weight, reduction_param_bias]))

        #net.f(InnerProduct(
        #    ip, self.pred_size, bottoms=[reduction],
        #    param_names=[ip_param_weight, ip_param_bias]))
        #net.f(Power(scale, scale=0.1, bottoms=[ip]))

        net.f(NumpyData(label, label_data))
        net.f(Wordvec(
                label_vec, self.config.att_hidden, len(MODULE_INDEX),
                bottoms=[label]))
        net.blobs[label_vec].reshape((batch_size, self.config.att_hidden, 1))
        net.f(Concat(concat, bottoms=[label_vec, reduction]))
        net.f(InnerProduct(
                ip, self.pred_size, bottoms=[concat],
                param_names=[ip_param_weight, ip_param_bias]))
        net.f(Power(scale, scale=0.1, bottoms=[ip]))

        return scale


class MeasureModule(Module):
    def forward(self, index, label_data, bottoms, image, dropout, apollo_net):
        assert len(bottoms) == 1
        mask = bottoms[0]

        net = apollo_net

        ip = "Measure_%d_ip" % index
        ip_param_weight = "Measure_ip_param_weight"
        ip_param_bias = "Measure_ip_param_bias"

        net.f(InnerProduct(
                ip, self.pred_size, bottoms=[mask],
                param_names=[ip_param_weight, ip_param_bias]))

        return ip


class Nmn:
    def __init__(self, index, modules, apollo_net):
        self.index = index
        self.apollo_net = apollo_net

        # TODO eww
        counter = [0]
        def number(tree):
            r = counter[0]
            counter[0] += 1
            return r
        numbered = util.tree_map(number, modules)
        assert util.flatten(numbered) == range(len(util.flatten(numbered)))

        def children(tree):
            if not isinstance(tree, tuple):
                return str(())
            # TODO nasty hack to make flatten behave right
            return str(tuple(c[0] if isinstance(c, tuple) else c for c in tree[1:]))
        child_annotated = util.tree_map(children, numbered)

        self.modules = util.flatten(modules)
        #print child_annotated
        #print util.flatten(child_annotated)
        #print [eval(c) for c in util.flatten(child_annotated)]
        self.children = [eval(c) for c in util.flatten(child_annotated)]

    def forward(self, label_data, image, dropout):
        flat_data = [util.flatten(d) for d in label_data]
        flat_data = np.asarray(flat_data)
        outputs = [None for i in range(len(self.modules))]
        for i in reversed(range(len(self.modules))):
            #print outputs, self.children
            bottoms = [outputs[j] for j in self.children[i]]
            assert None not in bottoms
            mod_index = self.index * 100 + i
            output = self.modules[i].forward(
                    mod_index, flat_data[:,i], bottoms, image, dropout,
                    self.apollo_net)
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
            self.nmns[modules] = Nmn(len(self.nmns), modules, self.apollo_net)
        return self.nmns[modules]


    def forward(self, layouts, layout_data, question_data, image_data, dropout, deterministic):

        # predict layout

        question_hidden = self.forward_question(question_data, dropout)
        layout_ids, layout_probs = \
                self.forward_layout(question_hidden, layouts, layout_data, deterministic)
        #layout_ids = [0 for i in range(len(layouts))]
        #layout_probs = [1 for i in range(len(layouts))]

        self.layout_ids = layout_ids
        self.layout_probs = layout_probs

        chosen_layouts = [ll[i] for ll, i in zip(layouts, layout_ids)]

        # prepare layout data

        module_layouts = list(set(l.modules for l in chosen_layouts))
        module_layout_choices = []
        default_labels = [None for i in range(len(module_layouts))]
        for layout in chosen_layouts:
            choice = module_layouts.index(layout.modules)
            module_layout_choices.append(choice)
            if default_labels[choice] is None:
                default_labels[choice] = layout.labels
        layout_label_data = []
        layout_mask = []
        for layout, choice in zip(chosen_layouts, module_layout_choices):
            labels_here = list(default_labels)
            labels_here[choice] = layout.labels
            layout_label_data.append(labels_here)
            mask_here = [0 for i in range(len(module_layouts))]
            mask_here[choice] = 1
            layout_mask.append(mask_here)
        #layout_label_data = np.asarray(layout_label_data)
        layout_mask = np.asarray(layout_mask)

        self.module_layout_choices = module_layout_choices

        # predict answer

        image = self.forward_image(image_data, dropout)

        nmn_hiddens = []
        for i in range(len(module_layouts)):
            module_layout = module_layouts[i]
            #label_data = layout_label_data[i]
            #print label_data
            label_data = [lld[i] for lld in layout_label_data]
            nmn = self.get_nmn(module_layout)
            nmn_hidden = nmn.forward(label_data, image, dropout)
            nmn_hiddens.append(nmn_hidden)

        # TODO mask and combine
        for h in nmn_hiddens:
            batch_size = self.apollo_net.blobs[h].shape[0]
            self.apollo_net.blobs[h].reshape(
                    (batch_size, self.config.pred_hidden, 1))
        if len(nmn_hiddens) == 1:
            concat_layer = nmn_hiddens[0]
        else:
            self.apollo_net.f(Concat("CHOOSE_concat", axis=2, bottoms=nmn_hiddens))
            concat_layer = "CHOOSE_concat"
        self.apollo_net.f(NumpyData("CHOOSE_mask", layout_mask))
        self.apollo_net.blobs["CHOOSE_mask"].reshape((batch_size, 1, len(module_layouts)))
        self.apollo_net.f(Tile("CHOOSE_tile_mask", axis=1, tiles=self.config.pred_hidden, bottoms=["CHOOSE_mask"]))
        self.apollo_net.f(Eltwise("CHOOSE_prod", "PROD", bottoms=["CHOOSE_tile_mask", concat_layer]))
        self.apollo_net.f(InnerProduct(
                "CHOOSE%d_sum" % len(module_layouts), 1, axis=2, bottoms=["CHOOSE_prod"],
                weight_filler=Filler("constant", 1),
                bias_filler=Filler("constant", 0),
                param_lr_mults=[0, 0]))
        self.apollo_net.blobs["CHOOSE%d_sum" % len(module_layouts)].reshape((batch_size, self.config.pred_hidden))
        nmn_hidden = "CHOOSE%d_sum" % len(module_layouts)

        self.prediction = self.forward_pred(question_hidden, nmn_hidden)

        self.prediction_data = self.apollo_net.blobs[self.prediction].data
        self.att_data = np.zeros((batch_size, 14, 14))
        #self.att_data = self.apollo_net.blobs["Attend_2_softmax"].data
        #self.att_data = self.att_data.reshape((-1, 14, 14))

    def forward_layout(self, question_hidden, layouts, layout_data, deterministic):
        net = self.apollo_net
        batch_size, n_layouts, n_features = layout_data.shape

        proj_question = "LAYOUT_proj_question"
        tile_question = "LAYOUT_tile%d_question" % n_layouts
        layout_feats = "LAYOUT_feats_%d"
        proj_layout = "LAYOUT_proj_layout_%d"
        #layout_word = "LAYOUT_word_%d"
        #layout_wordvec = "LAYOUT_wordvec_%d"
        concat = "LAYOUT_concat"
        sum = "LAYOUT_sum"
        relu = "LAYOUT_relu"
        pred = "LAYOUT_pred"
        softmax = "LAYOUT_softmax"

        net.f(InnerProduct(
                proj_question, self.config.layout_hidden,
                bottoms=[question_hidden]))
        net.blobs[proj_question].reshape(
                (batch_size, 1, self.config.layout_hidden))
        net.f(Tile(tile_question, axis=1, tiles=n_layouts,
                bottoms=[proj_question]))

        concat_bottoms = []
        for i in range(n_layouts):
            #net.f(NumpyData(layout_word % i, layout_data[:,i]))
            #net.f(Wordvec(
            #        layout_wordvec % i, self.config.layout_hidden,
            #        len(MODULE_INDEX), bottoms=[layout_word % i]))
            #net.blobs[layout_wordvec % i].reshape(
            #        (batch_size, 1, self.config.layout_hidden))
            #concat_bottoms.append(layout_wordvec % i)

            # TODO normalize these?
            net.f(NumpyData(layout_feats % i, layout_data[:,i,:]))
            net.f(InnerProduct(
                proj_layout % i, self.config.layout_hidden,
                bottoms=[layout_feats % i]))
            net.blobs[proj_layout % i].reshape(
                    (batch_size, 1, self.config.layout_hidden))
            concat_bottoms.append(proj_layout % i)

        if n_layouts > 1:
            net.f(Concat(concat, axis=1, bottoms=concat_bottoms))
            concat_layer = concat
        else:
            concat_layer = concat_bottoms[0]

        net.f(Eltwise(sum, "SUM", bottoms=[tile_question, concat_layer]))
        net.f(ReLU(relu, bottoms=[sum]))
        net.f(InnerProduct(pred, 1, axis=2, bottoms=[relu]))
        net.blobs[pred].reshape((batch_size, n_layouts))
        net.f(Softmax(softmax, bottoms=[pred]))

        probs = net.blobs[softmax].data

        layout_choices = []
        for i in range(len(layouts)):
            pr_here = probs[i,:len(layouts[i])].astype(np.float)
            pr_here /= np.sum(pr_here)
            if deterministic:
                choice = np.argmax(pr_here)
            else:
                choice = np.random.choice(pr_here.size, p=pr_here)
            layout_choices.append(choice)
            # TODO check to ensure choice is in bounds for this datum
        for i in range(batch_size - len(layouts)):
            layout_choices.append(0)

        return layout_choices, softmax

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
        question_dropout = "QUESTION_lstm_dropout"
        final_hidden = "QUESTION_lstm_final_hidden"

        prev_hidden = seed
        prev_mem = seed

        # TODO consolidate with module?
        if hasattr(self.config, "pred_hidden"):
            pred_size = self.config.pred_hidden
        else:
            pred_size = len(ANSWER_INDEX)

        if not hasattr(self.config, "lstm_hidden"):
            net.f(NumpyData(seed, np.zeros((batch_size, pred_size))))
            return seed

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

        if dropout:
            net.f(Dropout(question_dropout, 0.5, bottoms=[prev_hidden]))
            net.f(InnerProduct(final_hidden, pred_size, bottoms=[question_dropout]))
        else:
            net.f(InnerProduct(final_hidden, pred_size, bottoms=[prev_hidden]))

        return final_hidden

    def forward_pred(self, question_hidden, nmn_hidden):
        net = self.apollo_net

        sum = "PRED_sum"
        relu = "PRED_relu"
        ip = "PRED_ip"

        #net.f(ReLU(relu, bottoms=[question_hidden]))
        #net.f(InnerProduct(ip, len(ANSWER_INDEX), bottoms=[relu]))
        #return ip

        net.f(Eltwise(sum, "PROD", bottoms=[question_hidden, nmn_hidden]))

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
        pred_ans_probs += 1e-8
        pred_ans_log_probs = np.log(pred_ans_probs)
        pred_ans_log_probs[answer_data == UNK_ID] = 0

        self.cumulative_datum_losses -= pred_ans_log_probs

        return loss

    def reinforce_layout(self, losses):
        net = self.apollo_net

        choice_data = "REINFORCE_choice_data"
        loss_data = "REINFORCE_loss_data"
        index = "REINFORCE_index"
        weight = "REINFORCE_weight"
        reduction = "REINFORCE_reduction"
        zero = "REINFORCE_zero"
        loss = "REINFORCE_loss"

        #print self.layout_ids
        #print "\n".join([str(s) for s in zip(self.layout_ids, losses)])

        net.f(NumpyData(choice_data, self.layout_ids))
        net.f(NumpyData(loss_data, losses))
        net.f(Index(index, {}, bottoms=[self.layout_probs, choice_data]))
        net.f(Eltwise(weight, "PROD", bottoms=[index, loss_data]))

        net.f(NumpyData(zero, np.zeros(net.blobs[weight].shape)))
        net.f(EuclideanLoss(loss, bottoms=[weight, zero]))

        #net.f(AsLoss(loss, bottoms=[weight]))

        #from layers.reinforce import Reduction
        #net.f(Reduction(reduction, 0, bottoms=[weight], loss_weight=1.0))

    def reset(self):
        self.apollo_net.clear_forward()
        self.loss_counter = 0
        self.cumulative_datum_losses = np.zeros((self.opt_config.batch_size,))
        self.question_hidden = None

    def train(self):
        self.reinforce_layout(self.cumulative_datum_losses)
        self.apollo_net.backward()
        adadelta.update(self.apollo_net, self.opt_state, self.opt_config)

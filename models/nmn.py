from layers.reinforce import Index, AsLoss
from misc.indices import QUESTION_INDEX, MODULE_INDEX, ANSWER_INDEX, UNK_ID
from misc import util
from opt import adadelta

import apollocaffe
from apollocaffe.layers import *
import numpy as np

class Module(object):
    def __init__(self, config):
        self.config = config

        if hasattr(config, "pred_hidden"):
            self.pred_size = config.pred_hidden
        else:
            self.pred_size = len(ANSWER_INDEX)

    def forward(self, index, label_data, bottoms, features, rel_features, dropout, apollo_net):
        raise NotImplementedError()

    def __str__(self):
        return self.__class__.__name__

class AnswerAdaptor(Module):
    def __init__(self, config):
        super(AnswerAdaptor, self).__init__(config)
        self.mappings = dict()
        self.indices = dict()
        self.loaded_weights = False

    def register(self, key, mapping):
        if key in self.mappings:
            assert mapping == self.mappings[key]
            return self.indices[key]
        self.mappings[key] = mapping
        self.indices[key] = len(self.indices)
        return self.indices[key]

    def forward(self, index, label_data, bottoms, features, rel_features, dropout, apollo_net):
        assert len(bottoms) == 1
        input = bottoms[0]
        net = apollo_net
        assert self.config.att_normalization == "local"

        data = "AnswerAdaptor_%d_data" % index
        weights = "AnswerAdaptor_%d_weights" % index
        copy_input = "AnswerAdaptor_%d_copy_input" % index
        scalar = "AnswerAdaptor_%d_scalar" % index
        reduce = "AnswerAdaptor_%d_reduce" % index

        weight_param = "AnswerAdaptor_weight_param"
        reduce_param_w = "AnswerAdaptor_reduce_param_w"
        reduce_param_b = "AnswerAdaptor_reduce_param_b"

        batch_size = net.blobs[features].shape[0]
        # TODO correct
        n_mappings = 10
        n_inputs = net.blobs[features].shape[2]
        n_outputs = len(ANSWER_INDEX)

        net.f(NumpyData(data, label_data))
        net.f(Wordvec(
            weights, n_inputs * n_outputs, n_mappings, bottoms=[data],
            param_names=[weight_param], weight_filler=Filler("constant", 0),
            param_lr_mults=[0]))
        net.blobs[weights].reshape((batch_size, n_inputs, n_outputs))

        net.f(Power(copy_input, bottoms=[input]))
        net.blobs[copy_input].reshape((batch_size, n_inputs))
        net.f(Scalar(scalar, 0, bottoms=[weights, copy_input]))

        net.blobs[scalar].reshape((batch_size, n_inputs, n_outputs, 1))
        net.f(Convolution(
            reduce, (1, 1), 1, bottoms=[scalar],
            param_names=[reduce_param_w, reduce_param_b],
            weight_filler=Filler("constant", 1),
            bias_filler=Filler("constant", 0),
            param_lr_mults=[0,0]))
        net.blobs[reduce].reshape((batch_size, n_outputs))

        if not self.loaded_weights:
            for key, mapping in self.mappings.items():
                index = self.indices[key]
                for inp, out in mapping.items():
                    net.params[weight_param].data[0, index, 0, inp * n_outputs + out] = 1
            self.loaded_weights = True

        return reduce

class LookupModule(Module):
    def forward(self, index, label_data, bottoms, features, rel_features, dropout, apollo_net):
        assert len(bottoms) == 0
        net = apollo_net
        batch_size, channels, image_size, trailing = net.blobs[features].shape
        assert trailing == 1

        assert self.config.att_normalization == "local"
        data = np.zeros((batch_size, 1, image_size, trailing))
        for i in range(len(label_data)):
            data[i, :, label_data[i], ...] = 1

        lookup = "Lookup_%d_data" % index

        net.f(NumpyData(lookup, data))

        return lookup

class MLPFindModule(Module):
    def forward(self, index, label_data, bottoms, image, rel_features, dropout, 
            apollo_net):
        assert len(bottoms) == 0

        net = apollo_net

        batch_size, channels, image_size, trailing = net.blobs[image].shape
        assert trailing == 1

        proj_image = "Find_%d_proj_image" % index
        label = "Find_%d_label" % index
        label_vec = "Find_%d_label_vec" % index
        label_vec_dropout = "Find_%d_label_vec_dropout" % index
        tile = "Find_%d_tile" % index
        sum = "Find_%d_sum" % index
        relu = "Find_%d_relu" % index
        mask = "Find_%d_mask" % index
        softmax = "Find_%d_softmax" % index
        sigmoid = "Find_%d_sigmoid" % index
        copy = "Find_%d_copy" % index

        proj_image_param_weight = "Find_proj_image_param_weight"
        proj_image_param_bias = "Find_proj_image_param_bias"
        label_vec_param = "Find_label_vec_param"
        mask_param_weight = "Find_mask_param_weight"
        mask_param_bias = "Find_mask_param_bias"

        # compute attention mask

        net.f(Convolution(
                proj_image, (1, 1), self.config.att_hidden, bottoms=[image],
                param_names=[proj_image_param_weight, proj_image_param_bias]))

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

        net.f(Tile(tile, axis=2, tiles=image_size, bottoms=[label_vec_final]))
        net.f(Eltwise(sum, "SUM", bottoms=[proj_image, tile]))
        net.f(ReLU(relu, bottoms=[sum]))
        net.f(Convolution(mask, (1, 1), 1, bottoms=[relu],
                param_names=[mask_param_weight, mask_param_bias]))

        # TODO or defer?
        if self.config.att_normalization == "local":
            net.f(Sigmoid(sigmoid, bottoms=[mask]))
            prev = sigmoid
        elif self.config.att_normalization == "global":
            net.f(Softmax(softmax, bottoms=[mask]))
            prev = softmax
        net.f(Power(copy, bottoms=[prev]))
        net.blobs[copy].reshape((batch_size, 1, image_size, 1))

        return copy

class MultiplicativeFindModule(Module):
    def forward(self, index, label_data, bottoms, features, rel_features, dropout, apollo_net):
        assert len(bottoms) == 0

        net = apollo_net

        batch_size, channels, image_size, trailing = net.blobs[features].shape
        assert trailing == 1

        proj_image = "Find_%d_proj_image" % index
        label = "Find_%d_label" % index
        label_vec = "Find_%d_label_vec" % index
        label_vec_dropout = "Find_%d_label_vec_dropout" % index
        tile = "Find_%d_tile" % index
        prod = "Find_%d_prod" % index
        mask = "Find_%d_mask" % index
        sigmoid = "Find_%d_sigmoid" % index
        softmax = "Find_%d_softmax" % index
        copy = "Find_%d_copy" % index

        proj_image_param_weight = "Find_proj_image_param_weight"
        proj_image_param_bias = "Find_proj_image_param_bias"
        label_vec_param = "Find_label_vec_param"
        mask_param_weight = "Find_mask_param_weight"
        mask_param_bias = "Find_mask_param_bias"

        # compute attention mask

        net.f(NumpyData(label, label_data))
        net.f(Wordvec(
                label_vec, channels, len(MODULE_INDEX),
                bottoms=[label], param_names=[label_vec_param]))
        net.blobs[label_vec].reshape((batch_size, channels, 1, 1))
        if dropout:
            net.f(Dropout(label_vec_dropout, 0.5, bottoms=[label_vec]))
            label_vec_final = label_vec_dropout
        else:
            label_vec_final = label_vec

        net.f(Tile(tile, axis=2, tiles=image_size, bottoms=[label_vec_final]))

        net.f(Eltwise(prod, "PROD", bottoms=[features, tile]))
        net.f(Convolution(mask, (1, 1), 1, bottoms=[prod],
                param_names=[mask_param_weight, mask_param_bias],
                weight_filler=Filler("constant", 1),
                bias_filler=Filler("constant", 0),
                param_lr_mults=[0, 0]))

        if self.config.att_normalization == "local":
            net.f(Sigmoid(sigmoid, bottoms=[mask]))
            prev = sigmoid
        elif self.config.att_normalization == "global":
            net.f(Softmax(softmax, bottoms=[mask]))
            prev = softmax
        # TODO still WTF
        net.f(Power(copy, bottoms=[prev]))

        net.blobs[copy].reshape((batch_size, 1, image_size, 1))

        return copy

class RelateModule(Module):
    def forward(self, index, label_data, bottoms, features, rel_features,
            dropout, apollo_net):
        net = apollo_net
        batch_size, rel_channels, image_size, _ = net.blobs[rel_features].shape
        assert len(bottoms) == 1
        mask = bottoms[0]

        tile_mask_feats = "ReAttend_%d_tile_mask_feats" % index
        tile_mask_neighbors = "ReAttend_%d_tile_mask_neighbors" % index
        weight = "ReAttend_%d_weight" % index
        reduce = "ReAttend_%d_reduce" % index
        labels = "ReAttend_%d_labels" % index
        param = "ReAttend_%d_param" % index
        tile_param = "ReAttend_%d_tile_param" % index
        prod = "ReAttend_%d_prod" % index
        reduce2 = "ReAttend_%d_reduce2" % index
        sigmoid = "ReAttend_%d_sigmoid" % index
        softmax = "ReAttend_%d_softmax" % index
        copy = "ReAttend_%d_copy" % index

        reduce_param_weight = "ReAttend_reduce_param_weight"
        reduce_param_bias = "ReAttend_reduce_param_bias"
        wordvec_param = "ReAttend_wordvec_param"
        reduce2_param_weight = "ReAttend_reduce2_param_weight"
        reduce2_param_bias = "ReAttend_reduce2_param_bias"

        net.blobs[mask].reshape((batch_size, 1, 1, image_size))
        net.f(Tile(
                tile_mask_feats, axis=1, tiles=rel_channels, bottoms=[mask]))
        net.f(Tile(
                tile_mask_neighbors, axis=2, tiles=image_size,
                bottoms=[tile_mask_feats]))
        net.f(Eltwise(
                weight, "PROD", bottoms=[tile_mask_neighbors, rel_features]))
        net.f(InnerProduct(
                reduce, 1, axis=3, bottoms=[weight],
                weight_filler=Filler("constant", 1),
                bias_filler=Filler("constant", 0),
                param_lr_mults=[0, 0],
                param_names=[reduce_param_weight, reduce_param_bias]))

        net.f(NumpyData(labels, label_data))
        net.f(Wordvec(
                param, rel_channels, len(MODULE_INDEX), bottoms=[labels],
                param_names=[wordvec_param]))
        net.blobs[param].reshape((batch_size, rel_channels, 1, 1))
        net.f(Tile(tile_param, axis=2, tiles=image_size, bottoms=[param]))
        net.f(Eltwise(prod, "PROD", bottoms=[tile_param, reduce]))

        net.f(Convolution(
                reduce2, (1, 1), 1, bottoms=[prod],
                param_names=[reduce2_param_weight, reduce2_param_bias],
                weight_filler=Filler("constant", 1),
                bias_filler=Filler("constant", 0),
                param_lr_mults=[0, 0]))

        if self.config.att_normalization == "local":
                net.f(Sigmoid(sigmoid, bottoms=[reduce2]))
                prev = sigmoid
        elif self.config.att_normalization == "global":
                net.f(Softmax(softmax, bottoms=[reduce2]))
                prev = softmax
        # TODO still WTF
        net.f(Power(copy, bottoms=[prev]))

        return copy

class AndModule(Module):
    def forward(self, index, label_data, bottoms, features, rel_features, dropout, apollo_net):
        net = apollo_net
        assert len(bottoms) >= 2
        assert all(net.blobs[l].shape[1] == 1 for l in bottoms)

        prod = "And_%d_prod" % index

        net.f(Eltwise(prod, "PROD", bottoms=bottoms))

        return prod

class DescribeModule(Module):
    def forward(self, index, label_data, bottoms, features, rel_features, dropout, apollo_net):
        assert len(bottoms) == 1
        mask = bottoms[0]
        #assert self.config.att_normalization == "global"

        net = apollo_net

        batch_size, channels, image_size, _ = net.blobs[features].shape

        tile_mask = "Describe_%d_tile_mask" % index
        weight = "Describe_%d_weight" % index
        reduction = "Describe_%d_reduction" % index
        ip = "Describe_%d_ip" % index
        scale = "Describe_%d_scale" % index

        reduction_param_weight = "Describe_reduction_param_weight"
        reduction_param_bias = "Describe_reduction_param_bias"
        ip_param_weight = "Describe_ip_param_weight"
        ip_param_bias = "Describe_ip_param_bias"

        net.f(Tile(tile_mask, axis=1, tiles=channels, bottoms=[mask]))
        net.f(Eltwise(weight, "PROD", bottoms=[tile_mask, features]))
        net.f(InnerProduct(
            reduction, 1, axis=2, bottoms=[weight],
            weight_filler=Filler("constant", 1),
            bias_filler=Filler("constant", 0),
            param_lr_mults=[0, 0],
            param_names=[reduction_param_weight, reduction_param_bias]))
        net.f(InnerProduct(
            ip, self.pred_size, bottoms=[reduction],
            param_names=[ip_param_weight, ip_param_bias]))
        net.f(Power(scale, scale=0.001, bottoms=[ip]))

        return scale

class ExistsModule(Module):
    def forward(self, index, label_data, bottoms, features, rel_features, dropout, apollo_net):
        assert len(bottoms) == 1
        mask = bottoms[0]

        net = apollo_net

        reduce = "Exists_%d_reduce" % index
        ip = "Exists_%d_ip" % index

        reduce_param_weight = "Exists_reduce_param_weight"
        reduce_param_bias = "Exists_reduce_param_bias"
        ip_param_weight = "Exists_ip_param_weight"
        ip_param_bias = "Exists_ip_param_bias"

        net.f(Pooling(reduce, kernel_h=10, kernel_w=1, bottoms=[mask]))

        net.f(InnerProduct(
            ip, len(ANSWER_INDEX), bottoms=[reduce],
            param_names=[ip_param_weight, ip_param_bias],
            weight_filler=Filler("constant", 0),
            bias_filler=Filler("constant", 0),
            param_lr_mults=[0, 0]))
        net.params[ip_param_weight].data[ANSWER_INDEX["yes"],0] = 1
        net.params[ip_param_weight].data[ANSWER_INDEX["no"],0] = -1
        net.params[ip_param_bias].data[ANSWER_INDEX["no"]] = 1

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
        self.children = [eval(c) for c in util.flatten(child_annotated)]

    def forward(self, label_data, features, rel_features, dropout):

        flat_data = [util.flatten(d) for d in label_data]
        flat_data = np.asarray(flat_data)
        outputs = [None for i in range(len(self.modules))]
        for i in reversed(range(len(self.modules))):
            bottoms = [outputs[j] for j in self.children[i]]
            assert None not in bottoms
            mod_index = self.index * 100 + i
            output = self.modules[i].forward(
                    mod_index, flat_data[:,i], bottoms, features, rel_features,
                    dropout, self.apollo_net)
            outputs[i] = output

        self.outputs = outputs
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


    def forward(self, layouts, layout_data, question_data, features_data,
            rel_features_data, dropout, deterministic):

        # predict layout

        question_hidden = self.forward_question(question_data, dropout)
        layout_ids, layout_probs = \
                self.forward_layout(question_hidden, layouts, layout_data,
                        deterministic)

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
        layout_mask = np.asarray(layout_mask)

        self.module_layout_choices = module_layout_choices

        # predict answer

        features = self.forward_features(0, features_data, dropout)
        if rel_features_data is not None:
            rel_features = self.forward_features(1, rel_features_data, dropout)
        else:
            rel_features = None

        nmn_hiddens = []
        nmn_outputs = []
        for i in range(len(module_layouts)):
            module_layout = module_layouts[i]
            label_data = [lld[i] for lld in layout_label_data]
            nmn = self.get_nmn(module_layout)
            nmn_hidden = nmn.forward(label_data, features, rel_features, dropout)
            nmn_hiddens.append(nmn_hidden)
            nmn_outputs.append(nmn.outputs)

        chosen_hidden = self.forward_choice(module_layouts, layout_mask, nmn_hiddens)
        self.prediction = self.forward_pred(question_hidden, chosen_hidden)

        batch_size = self.apollo_net.blobs[nmn_hiddens[0]].shape[0]
        self.prediction_data = self.apollo_net.blobs[self.prediction].data
        self.att_data = np.zeros((batch_size, 14, 14))

    def forward_choice(self, module_layouts, layout_mask, nmn_hiddens):
        net = self.apollo_net

        concat = "CHOOSE_concat"
        mask = "CHOOSE_mask"
        prod = "CHOOSE_prod"
        tile_mask = "CHOOSE_tile_mask"
        sum = "CHOOSE%d_sum" % len(module_layouts)

        batch_size = self.apollo_net.blobs[nmn_hiddens[0]].shape[0]

        output_hidden = self.config.pred_hidden \
                if hasattr(self.config, "pred_hidden") \
                else len(ANSWER_INDEX)
        for h in nmn_hiddens:
            self.apollo_net.blobs[h].reshape((batch_size, output_hidden, 1))
        if len(nmn_hiddens) == 1:
            concat_layer = nmn_hiddens[0]
        else:
            self.apollo_net.f(Concat(concat, axis=2, bottoms=nmn_hiddens))
            concat_layer = concat

        self.apollo_net.f(NumpyData(mask, layout_mask))
        self.apollo_net.blobs[mask].reshape(
                (batch_size, 1, len(module_layouts)))
        self.apollo_net.f(Tile(
                tile_mask, axis=1, tiles=output_hidden, bottoms=[mask]))
        self.apollo_net.f(Eltwise(
                prod, "PROD", bottoms=[tile_mask, concat_layer]))
        self.apollo_net.f(InnerProduct(
                sum, 1, axis=2, bottoms=[prod],
                weight_filler=Filler("constant", 1),
                bias_filler=Filler("constant", 0),
                param_lr_mults=[0, 0]))
        self.apollo_net.blobs[sum].reshape((batch_size, output_hidden))

        return sum

    def forward_layout(self, question_hidden, layouts, layout_data,
            deterministic=False):
        net = self.apollo_net
        batch_size, n_layouts, n_features = layout_data.shape

        proj_question = "LAYOUT_proj_question"
        tile_question = "LAYOUT_tile%d_question" % n_layouts
        layout_feats = "LAYOUT_feats_%d"
        proj_layout = "LAYOUT_proj_layout_%d"
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
            # TODO normalize these?
            net.f(NumpyData(layout_feats % i, layout_data[:,i,:]))
            net.f(InnerProduct(proj_layout % i, self.config.layout_hidden,
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
        for i in range(batch_size - len(layouts)):
            layout_choices.append(0)

        return layout_choices, softmax

    def forward_features(self, index, feature_data, dropout):
        batch_size = feature_data.shape[0]
        channels = feature_data.shape[1]
        image_size = feature_data.shape[2] * feature_data.shape[3]

        net = self.apollo_net

        features = "FEATURES_%d_data" % index
        features_dropout = "FEATURES_%d_dropout" % index

        if features not in net.blobs:
            net.f(DummyData(features, feature_data.shape))
        net.blobs[features].data[...] = feature_data

        if dropout:
            net.f(Dropout(features_dropout, 0.5, bottoms=[features]))
            return features_dropout
        else:
            return features

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
            net.f(Dropout(question_dropout, 0.5, bottoms=[prev_hidden]))
            net.f(InnerProduct(final_hidden, pred_size, bottoms=[question_dropout]))
        else:
            net.f(InnerProduct(final_hidden, pred_size, bottoms=[prev_hidden]))

        return final_hidden

    def forward_pred(self, question_hidden, nmn_hidden):
        net = self.apollo_net

        relu = "PRED_relu"
        ip = "PRED_ip"

        if self.config.combine_question:
            sum = "PRED_sum"
            net.f(Eltwise(sum, "SUM", bottoms=[question_hidden, nmn_hidden]))
        else:
            sum = nmn_hidden

        if hasattr(self.config, "pred_hidden"):
            net.f(ReLU(relu, bottoms=[sum]))
            net.f(InnerProduct(ip, len(ANSWER_INDEX), bottoms=[relu]))
            return ip
        else:
            return sum

    def loss(self, answer_data, multiclass=False):
        net = self.apollo_net

        target = "PRED_target_%d" % self.loss_counter
        loss = "PRED_loss_%d" % self.loss_counter
        datum_loss = "PRED_datum_loss_%d" % self.loss_counter
        self.loss_counter += 1

        if multiclass:
            net.f(NumpyData(target, answer_data))
            acc_loss = net.f(EuclideanLoss(
                loss, bottoms=[self.prediction, target]))

            pred_probs = net.blobs[self.prediction].data
            batch_size = pred_probs.shape[0]
            pred_ans_probs = np.sum(np.abs(answer_data - pred_probs) ** 2, axis=1)
            # TODO
            pred_ans_log_probs = pred_ans_probs

        else:
            net.f(NumpyData(target, answer_data))
            acc_loss = net.f(SoftmaxWithLoss(
                loss, bottoms=[self.prediction, target], ignore_label=UNK_ID))

            net.f(Softmax(datum_loss, bottoms=[self.prediction]))

            pred_probs = net.blobs[datum_loss].data
            batch_size = pred_probs.shape[0]
            pred_ans_probs = pred_probs[np.arange(batch_size), answer_data.astype(np.int)]
            pred_ans_log_probs = np.log(pred_ans_probs)
            pred_ans_log_probs[answer_data == UNK_ID] = 0

        self.cumulative_datum_losses += pred_ans_log_probs

        return acc_loss

    def reinforce_layout(self, losses):
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

        net.f(AsLoss(loss, bottoms=[weight]))

    def reset(self):
        self.apollo_net.clear_forward()
        self.loss_counter = 0
        self.cumulative_datum_losses = np.zeros((self.opt_config.batch_size,))
        self.question_hidden = None

    def train(self):
        self.reinforce_layout(self.cumulative_datum_losses)
        self.apollo_net.backward()
        adadelta.update(self.apollo_net, self.opt_state, self.opt_config)

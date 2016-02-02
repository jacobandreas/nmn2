#!/usr/bin/env python2

# check profiler
if not isinstance(__builtins__, dict) or "profile" not in __builtins__:
    __builtins__.__dict__["profile"] = lambda x: x

from misc import util
from misc.indices import QUESTION_INDEX, ANSWER_INDEX, MODULE_INDEX, MODULE_TYPE_INDEX, \
        NULL, NULL_ID, UNK_ID
from misc.visualizer import visualizer
import models
from models.nmn import MLPFindModule, MultiplicativeFindModule
import tasks

import apollocaffe
import argparse
import json
import logging.config
import random
import numpy as np
import yaml

def main():
    config = configure()
    task = tasks.load_task(config)
    model = models.build_model(config.model, config.opt)

    for i_epoch in range(config.opt.iters):

        train_loss, train_acc, _ = \
                do_iter(task.train, model, config, train=True)
        val_loss, val_acc, val_predictions = \
                do_iter(task.val, model, config, vis=True)
        test_loss, test_acc, test_predictions = \
                do_iter(task.test, model, config)

        logging.info(
                "%5d  |  %8.3f  %8.3f  %8.3f  |  %8.3f  %8.3f  %8.3f",
                i_epoch,
                train_loss, val_loss, test_loss,
                train_acc, val_acc, test_acc)

        with open("logs/val_predictions_%d.json" % i_epoch, "w") as pred_f:
            print >>pred_f, json.dumps(val_predictions)

        #with open("logs/test_predictions_%d.json" % i_epoch, "w") as pred_f:
        #    print >>pred_f, json.dumps(test_predictions)

def configure():
    apollocaffe.set_random_seed(0)
    np.random.seed(0)
    random.seed(0)

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
            "-c", "--config", dest="config", required=True,
            help="model configuration file")
    arg_parser.add_argument(
            "-l", "--log-config", dest="log_config", default="config/log.yml",
            help="log configuration file")

    args = arg_parser.parse_args()
    config_name = args.config.split("/")[-1].split(".")[0]

    with open(args.log_config) as log_config_f:
        log_filename = "logs/%s.log" % config_name
        log_config = yaml.load(log_config_f)
        log_config["handlers"]["fileHandler"]["filename"] = log_filename
        logging.config.dictConfig(log_config)

    with open(args.config) as config_f:
        config = util.Struct(**yaml.load(config_f))

    assert not hasattr(config, "name")
    config.name = config_name

    return config

def do_iter(task_set, model, config, train=False, vis=False):
    loss = 0.0
    acc = 0.0
    predictions = []
    n_batches = 0

    # sort first to guarantee deterministic behavior with fixed seed
    data = list(sorted(task_set.data))
    np.random.shuffle(data)

    if vis:
        visualizer.begin(config.name, 100)

    for batch_start in range(0, len(data), config.opt.batch_size):
        batch_end = batch_start + config.opt.batch_size
        batch_data = data[batch_start:batch_end]

        batch_loss, batch_acc, batch_preds = do_batch(
                batch_data, model, config, train, vis)

        loss += batch_loss
        acc += batch_acc
        predictions += batch_preds
        n_batches += 1

        if vis:
            visualize(batch_data, model)

    if vis:
        visualizer.end()

    if n_batches == 0:
        return 0, 0, dict()
    assert len(predictions) == len(data)
    loss /= n_batches
    acc /= n_batches
    return loss, acc, predictions

def do_batch(data, model, config, train, vis):
    predictions = forward(data, model, config, train, vis)
    answer_loss = backward(data, model, config, train, vis)
    acc = compute_acc(predictions, data, config)

    return answer_loss, acc, predictions

# TODO this is ugly and belongs somewhere else
def featurize_layouts(datum, max_layouts):
    # TODO pre-fill module type index
    layout_reprs = np.zeros((max_layouts, len(MODULE_INDEX) + 7))
    for i_layout in range(len(datum.layouts)):
        layout = datum.layouts[i_layout]
        labels = util.flatten(layout.labels)
        modules = util.flatten(layout.modules)
        for i_mod in range(len(modules)):
            if isinstance(modules[i_mod], MLPFindModule) or isinstance(modules[i_mod], MultiplicativeFindModule):
                layout_reprs[i_layout, labels[i_mod]] += 1
            mt = MODULE_TYPE_INDEX.index(modules[i_mod])
            layout_reprs[i_layout, len(MODULE_INDEX) + mt] += 1
    return layout_reprs

def forward(data, model, config, train, vis):
    model.reset()

    # load batch data
    max_len = max(len(d.question) for d in data)
    max_layouts = max(len(d.layouts) for d in data)
    channels, size, trailing = data[0].load_features().shape
    assert trailing == 1
    has_rel_features = data[0].load_rel_features() is not None
    if has_rel_features:
        rel_channels, size_1, size_2 = data[0].load_rel_features().shape
        assert size_1 == size_2 == size
    questions = np.ones((config.opt.batch_size, max_len)) * NULL_ID
    features = np.zeros((config.opt.batch_size, channels, size, 1))
    if has_rel_features:
        rel_features = np.zeros((config.opt.batch_size, rel_channels, size, size))
    else:
        rel_features = None
    layout_reprs = np.zeros(
            (config.opt.batch_size, max_layouts, len(MODULE_INDEX) + 7))
    for i, datum in enumerate(data):
        questions[i, max_len-len(datum.question):] = datum.question
        features[i, ...] = datum.load_features()
        if has_rel_features:
            rel_features[i, ...] = datum.load_rel_features()
        layout_reprs[i, ...] = featurize_layouts(datum, max_layouts)
    layouts = [d.layouts for d in data]

    # apply model
    model.forward(
            layouts, layout_reprs, questions, features, rel_features, 
            dropout=(train and config.opt.dropout), deterministic=not train)

    # extract predictions
    if config.opt.multiclass:
        pred_words = []
        for i in range(model.prediction_data.shape[0]):
            preds = model.prediction_data[i, :]
            chosen = np.where(preds > 0.5)[0]
            pred_words.append(set(ANSWER_INDEX.get(w) for w in chosen))
    else:
        pred_ids = np.argmax(model.prediction_data, axis=1)
        pred_words = [ANSWER_INDEX.get(w) for w in pred_ids]
    predictions = list()
    for i in range(len(data)):
        qid = data[i].id
        answer = pred_words[i]
        predictions.append({"question_id": qid, "answer": answer})

    return predictions

def backward(data, model, config, train, vis):
    n_answers = len(data[0].answers)
    loss = 0

    for i in range(n_answers):
        if config.opt.multiclass:
            output_i = np.zeros((config.opt.batch_size, len(ANSWER_INDEX)))
            for i_datum, datum in enumerate(data):
                for answer in datum.answers[i]:
                    output_i[i_datum, answer] = 1
        else:
            output_i = UNK_ID * np.ones(config.opt.batch_size)
            output_i[:len(data)] = \
                    np.asarray([d.answers[i] for d in data])
        loss += model.loss(output_i, multiclass=config.opt.multiclass)

    if train:
        model.train()

    return loss

def visualize(batch_data, model):
    i_datum = 0
    #mod_layout_choice = model.module_layout_choices[i_datum]
    #print model.apollo_net.blobs.keys()
    #att_blob_name = "Find_%d_softmax" % (mod_layout_choice * 100 + 1)
    #
    datum = batch_data[i_datum]
    question = " ".join([QUESTION_INDEX.get(w) for w in datum.question[1:-1]]),
    preds = model.prediction_data[i_datum,:]
    top = np.argsort(preds)[-5:]
    top_answers = reversed([ANSWER_INDEX.get(p) for p in top])
    #att_data = model.apollo_net.blobs[att_blob_name].data[i_datum,...]
    #att_data = att_data.reshape((14, 14))
    att_data = np.zeros((14, 14))
    chosen_parse = datum.parses[model.layout_ids[i_datum]]
    
    fields = [
        question,
        str(chosen_parse),
        "<img src='../../%s'>" % datum.image_path,
        att_data,
        ", ".join(top_answers),
        ", ".join([ANSWER_INDEX.get(a) for a in datum.answers])
    ]
    visualizer.show(fields)

def compute_acc(predictions, data, config):
    score = 0.0
    for prediction, datum in zip(predictions, data):
        pred_answer = prediction["answer"]
        if config.opt.multiclass:
            answers = [set(ANSWER_INDEX.get(aa) for aa in a) for a in datum.answers]
        else:
            answers = [ANSWER_INDEX.get(a) for a in datum.answers]

        matching_answers = [a for a in answers if a == pred_answer]
        if len(answers) == 1:
            score += len(matching_answers)
        else:
            score += min(len(matching_answers) / 3.0, 1.0)
    score /= len(data)
    return score

if __name__ == "__main__":
    main()

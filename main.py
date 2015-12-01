#!/usr/bin/env python2

# check profiler
if not isinstance(__builtins__, dict) or "profile" not in __builtins__:
    __builtins__.__dict__["profile"] = lambda x: x

from misc import util
from misc.indices import QUESTION_INDEX, ANSWER_INDEX, NULL, UNK_ID
from misc.visualizer import visualizer
import models
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

    for t in range(config.opt.iters):
        train_loss, train_acc, _ = \
                do_iter(task.train, model, config, train=True)
        val_loss, val_acc, val_predictions = \
                do_iter(task.val, model, config, vis=True)
        test_loss, test_acc, _ = \
                do_iter(task.test, model, config)

        logging.info(
                "%5d  |  %2.3f  %2.3f  %2.3f  |  %2.3f  %2.3f  %2.3f", 
                t, 
                train_loss, val_loss, test_loss,
                train_acc, val_acc, test_acc)

        with open("logs/val_predictions_%d.json" % t, "w") as pred_f:
            print >>pred_f, json.dumps(val_predictions.values())

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


def do_iter(data, model, config, train=False, vis=False):
    loss = 0.0
    acc = 0.0
    batches = 0
    layout_types = list(data.by_layout_type.keys())
    np.random.shuffle(layout_types)

    predictions = dict()

    if vis:
        visualizer.begin(config.name, 100)

    for layout_type in layout_types:
        key_data = list(data.by_layout_type[layout_type])
        np.random.shuffle(key_data)
        for batch_start in range(0, len(key_data), config.opt.batch_size):
            batch_end = batch_start + config.opt.batch_size
            batch_data = key_data[batch_start:batch_end]
            batches += 1

            channels, width, height = batch_data[0].load_image().shape
            length = max(len(d.question) for d in batch_data)
            n_answers = len(batch_data[0].answers)
            batch_images = np.zeros(
                    (config.opt.batch_size, channels, width, height))
            batch_questions = np.ones(
                    (config.opt.batch_size, length)) * QUESTION_INDEX[NULL]
            for i, datum in enumerate(batch_data):
                image = datum.load_image()
                d_length = len(datum.question)
                batch_images[i,...] = image
                batch_questions[i,length-d_length:] = datum.question

            batch_layout_labels = [d.layout.labels for d in batch_data]
            batch_layout_labels = util.tree_zip(*batch_layout_labels)
            model.forward(
                    layout_type, batch_layout_labels, batch_questions, 
                    batch_images, dropout=train)
            for i in range(n_answers):
                batch_output_i = UNK_ID * np.ones(config.opt.batch_size)
                batch_output_i[:len(batch_data)] = \
                        np.asarray([d.answers[i] for d in batch_data])
                loss += model.loss(batch_output_i)
                acc += compute_acc(model.prediction_data, batch_output_i)
            
            pred_ids = np.argmax(model.prediction_data, axis=1)
            pred_words = [ANSWER_INDEX.get(w) for w in pred_ids]
            for i in range(len(batch_data)):
                qid = batch_data[i].id
                answer = pred_words[i]
                predictions[qid] = {"question_id": qid, "answer": answer}

            if train:
                model.train()

            if vis:
                datum = batch_data[0]
                preds = model.prediction_data[0,:]
                top = np.argsort(preds)[-5:]
                top_answers = reversed([ANSWER_INDEX.get(p) for p in top])
                fields = [
                        " ".join([QUESTION_INDEX.get(w) for w in datum.question[1:-1]]),
                    "<img src='../../%s'>" % datum.image_path,
                    model.att_data[0,...],
                    ", ".join(top_answers),
                    ", ".join([ANSWER_INDEX.get(a) for a in datum.answers])
                ]
                visualizer.show(fields)

    if vis:
        visualizer.end()

    if batches == 0:
        return 0, 0, dict()
    return loss / batches, acc / batches, predictions

def compute_acc(pred, target):
    pred_ids = np.argmax(pred, axis=1)
    return np.mean(pred_ids == target)

if __name__ == "__main__":
    main()

#!/usr/bin/env python2

import corpus
from nmn import NMN
from lstm import LSTM
from input_types import *
from output_types import *
from shape import *

from collections import defaultdict
import logging, logging.config
import numpy as np
import theano
import yaml

#theano.config.optimizer_including = "local_log_softmax"
#theano.config.optimizer = "fast_compile"
#theano.config.optimizer = "stabilization"

LOG_CONFIG = "log.yaml"
EXPERIMENT_CONFIG = "config.yaml"

def train_iter(train_data_by_query, train_queries):
    epoch_train_ll = 0.
    epoch_train_acc = 0.

    for query in train_queries:
        data = train_data_by_query[query]
        np.random.shuffle(data)
        data = data[:256]
        batch_inputs = np.asarray([datum.input_ for datum in data], dtype=input_type.dtype)
        batch_outputs = np.asarray([datum.output for datum in data], dtype=output_type.dtype)
    
        train_ll = model.train(query, batch_inputs, batch_outputs, return_norm=False)
        batch_pred = model.predict(query, batch_inputs)
        epoch_train_ll += train_ll
        epoch_train_acc += 1. * sum(np.equal(batch_pred, batch_outputs)) / len(data)
    
    epoch_train_ll /= len(train_queries)
    epoch_train_acc /= len(train_queries)

    return epoch_train_ll, epoch_train_acc

def eval_iter(val_data_by_query):
    epoch_val_ll = 0.
    epoch_val_acc = 0.

    epoch_acc_by_size = defaultdict(lambda: 0)
    epoch_counts_by_size = defaultdict(lambda: 0)
    for query, data in val_data_by_query.items():
        data = data[:256]
        batch_inputs = np.asarray([datum.input_ for datum in data], dtype=input_type.dtype)
        batch_outputs = np.asarray([datum.output for datum in data], dtype=output_type.dtype)
        val_ll = model.loss(query, batch_inputs, batch_outputs)
        batch_pred = model.predict(query, batch_inputs)
        epoch_val_ll += val_ll
        acc_here = 1. * sum(np.equal(batch_pred, batch_outputs)) / len(data)
        epoch_val_acc += acc_here

        epoch_val_ll /= len(val_data_by_query)
        epoch_val_acc /= len(val_data_by_query)

    return epoch_val_ll, epoch_val_acc

if __name__ == "__main__":

    with open(LOG_CONFIG) as log_config_f:
        logging.config.dictConfig(yaml.load(log_config_f))

    with open(EXPERIMENT_CONFIG) as experiment_config_f:
        config = yaml.load(experiment_config_f)["experiment"]

    train_data = corpus.load(config["corpus"], "train.%s" % config["train_size"])
    logging.info("loaded train.%s", config["train_size"])
    val_data = corpus.load(config["corpus"], "val")
    logging.info("loaded val")
    test_data = corpus.load(config["corpus"], "test")
    logging.info("loaded test")

    input_type = eval(config["input_type"])
    output_type = eval(config["output_type"])

    model_params = config["model_params"]
    model = globals()[config["model"]](input_type, output_type, model_params)

    train_data_by_query = defaultdict(list)
    for datum in train_data:
        train_data_by_query[datum.query].append(datum) 

    val_data_by_query = defaultdict(list)
    for datum in val_data:
        val_data_by_query[datum.query].append(datum)

    test_data_by_query = defaultdict(list)
    for datum in test_data:
        test_data_by_query[datum.query].append(datum)

    train_queries = list(train_data_by_query.keys())

    for i in range(40000):
        np.random.shuffle(train_queries)
        epoch_train_ll, epoch_train_acc = train_iter(train_data_by_query,
                train_queries)

        epoch_val_ll, epoch_val_acc = eval_iter(val_data_by_query)
        epoch_test_ll, epoch_test_acc = eval_iter(test_data_by_query)

        logging.info("%0.4f\t%0.4f\t|\t%0.4f\t%0.4f", epoch_train_ll,
                epoch_val_ll, epoch_train_acc, epoch_val_acc)
        logging.info("%s", epoch_acc_by_size)
        logging.info("%s", epoch_counts_by_size)

        if i % 10 == 0:
            model.serialize("model.txt")

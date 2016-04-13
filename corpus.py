#!/usr/bin/env python2

from util import Index
from data.util import pp

import numpy as np
import sexpdata

class QueryDatum:
    def __init__(self, query, input_, output):
        self.query = query
        self.input_ = input_
        self.output = output

def extract_query(sexp_query):
    if isinstance(sexp_query, sexpdata.Symbol):
        return sexp_query.value()
    return tuple(extract_query(q) for q in sexp_query)

def parse_query(query):
    parsed = sexpdata.loads(query)
    extracted = extract_query(parsed)
    return extracted

def load_math(set_name):
    data = []
    with open("data/math/%s.query" % set_name) as query_f, \
             open("data/math/%s.input" % set_name) as input_f, \
             open("data/math/%s.output" % set_name) as output_f:
        for query_str, input_str, output_str in zip(query_f, input_f, output_f):
            query = parse_query(query_str.strip())
            input_ = np.asarray([float(f) for f in input_str.strip().split()])
            output = np.asarray([float(f) for f in output_str.strip().split()])
            data.append(QueryDatum(query, input_, output))
    return data

def load_shapes(set_name):
    data = []
    index = {"true": 1, "false": 0}
    with open("data/shapes/%s.query" % set_name) as query_f, \
         open("data/shapes/%s.output" % set_name) as output_f:
        inputs = np.load("data/shapes/%s.input.npy" % set_name)
        for query_str, i_input, output_str in zip(query_f, range(inputs.shape[0]), output_f):
            query = parse_query(query_str.strip())
            inp = inputs[i_input,:,:,:].transpose((2,0,1)) / 255.
            output = index[output_str.strip()]
            data.append(QueryDatum(query, inp, output))
    return data

def load(corpus_name, set_name):
    if corpus_name == 'shapes':
        return load_shapes(set_name)
    else:
        assert False

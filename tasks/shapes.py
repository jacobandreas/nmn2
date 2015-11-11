#!/usr/bin/env python2

from datum import Datum, Layout
from indices import STRING_INDEX, LAYOUT_INDEX, ANSWER_INDEX, UNK
from parse import parse_tree
from models.modules import *

from collections import defaultdict
import logging
import numpy as np
import os

STRING_FILE = "data/shapes/%s.txt"
PARSE_FILE = "data/shapes/%s.query"
ANN_FILE = "data/shapes/%s.output"
IMAGE_FILE = "data/shapes/%s.input.npy"

def parse_to_layout(parse):
    return Layout(*parse_to_layout_helper(parse))

def parse_to_layout_helper(parse):
    if isinstance(parse, str):
        #return (DetectModule, LAYOUT_INDEX.index(parse))
        return (DetectModule, LAYOUT_INDEX.get_or_else(parse, LAYOUT_INDEX[UNK]))
    else:
        head = parse[0]
        #head_idx = LAYOUT_INDEX.index(parse)
        head_idx = LAYOUT_INDEX.get_or_else(head, LAYOUT_INDEX[UNK])
        if head == "is":
            mod_head = CombAnswerModule
        elif head in ("color", "shape"):
            mod_head = AttAnswerModule
        else:
            mod_head = RedetectModule
        #else:
        #    if head == "count":
        #        mod_head = DenseAnswerModule
        #    #elif head == "where":
        #    #    mod_head = AttAnswerModuleCopy
        #    else:
        #        mod_head = AttAnswerModule

        below = [parse_to_layout_helper(child) for child in parse[1:]]
        mods_below, indices_below = zip(*below)
        return (mod_head,) + tuple(mods_below), (head_idx,) + tuple(indices_below)

class ShapesDatum(Datum):
    def __init__(self, string, layout, image, answer):
        self.string = string
        self.layout = layout
        self.image = image
        self.answer = answer
        self.outputs = [answer]

    def load_input(self):
        return self.image

class ShapesTask:
    def __init__(self, config):
        self.train = ShapesTaskSet(config, "train.large")
        self.val = ShapesTaskSet(config, "val")
        self.test = ShapesTaskSet(config, "test")

class ShapesTaskSet:
    def __init__(self, config, set_name):
        self.config = config
        
        data = set()
        data_by_layout_type = defaultdict(list)
        data_by_string_length = defaultdict(list)
        data_by_layout_and_length = defaultdict(list)

        images = np.load(IMAGE_FILE % set_name)

        with open(STRING_FILE % set_name) as question_f, \
             open(PARSE_FILE % set_name) as parse_f, \
             open(ANN_FILE % set_name) as ann_f:

            i = 0
            for question, parse_str, answer in zip(question_f, parse_f, ann_f):
        
                question = question.strip().split()
                question = ["<s>"] + question + ["</s>"]
                question = [STRING_INDEX.index(w) for w in question]
                parse = parse_tree(parse_str.strip())
                layout = parse_to_layout(parse)

                answer = ANSWER_INDEX.index(answer)
                datum = ShapesDatum(question, layout, images[i,...], answer)
                datum.raw_query = parse_str

                data.add(datum)
                data_by_layout_type[datum.layout.modules].append(datum)
                data_by_string_length[len(datum.string)].append(datum)
                data_by_layout_and_length[(datum.layout.modules, len(datum.string))].append(datum)

                i += 1

        self.data = data
        self.by_layout_type = data_by_layout_type
        self.by_string_length = data_by_string_length
        self.by_layout_and_length = data_by_layout_and_length

        logging.info("%s:", set_name.upper())
        logging.info("%s items", len(self.data))
        logging.info("%s words", len(STRING_INDEX))
        logging.info("%s functions", len(LAYOUT_INDEX))
        logging.info("%s answers", len(ANSWER_INDEX))
        logging.info("%s layouts", len(self.by_layout_type.keys()))
        logging.info("")

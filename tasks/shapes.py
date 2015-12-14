#!/usr/bin/env python2

from misc.datum import Datum, Layout
from misc.indices import QUESTION_INDEX, MODULE_INDEX, ANSWER_INDEX
from misc.parse import parse_tree
#from models.modules import *
from models.nmn import AttendModule, ClassifyModule, MeasureModule, \
        CombineModule

from collections import defaultdict
import logging
import numpy as np
import os

#QUESTION_FILE = "data/shapes/%s.txt"
QUESTION_FILE = "data/shapes/%s.query"
PARSE_FILE = "data/shapes/%s.query"
ANN_FILE = "data/shapes/%s.output"
IMAGE_FILE = "data/shapes/conv/%s.%s.png.npz"
RAW_IMAGE_FILE = "data/shapes/raw/%s.%s.png"

def parse_to_layout_helper(parse, config, modules):
    if isinstance(parse, str):
        return modules["attend"], MODULE_INDEX.index(parse)
    head = parse[0]
    below = [parse_to_layout_helper(c, config, modules) for c in parse[1:]]
    modules_below, labels_below = zip(*below)
    modules_below = tuple(modules_below)
    labels_below = tuple(labels_below)
    if head == "and":
        module_head = modules["combine"]
    elif head == "is":
        module_head = modules["measure"]
    else:
        module_head = modules["classify"]
    label_head = MODULE_INDEX.index(head)
    modules_here = (module_head,) + modules_below
    labels_here = (label_head,) + labels_below
    return modules_here, labels_here

def parse_to_layout(parse, config, modules):
    modules, indices = parse_to_layout_helper(parse, config, modules)
    return Layout(modules, indices)

class ShapesDatum(Datum):
    def __init__(self, id, string, layout, image, answer, image_path):
        self.id = id
        self.string = string
        self.layouts = [layout]
        self.image = image
        self.answer = answer
        self.answers = [answer]
        self.image_path = image_path

    def load_image(self):
        return self.image

class ShapesTask:
    def __init__(self, config):

        modules = {
            "attend": AttendModule(config.model),
            "classify": ClassifyModule(config.model),
            "measure": MeasureModule(config.model),
            "combine": CombineModule(config.model)
        }

        self.train = ShapesTaskSet(config, "train", modules)
        self.val = ShapesTaskSet(config, "val", modules)
        self.test = ShapesTaskSet(config, "test", modules)

class ShapesTaskSet:
    def __init__(self, config, set_name, modules):
        self.config = config

        data = set()
        data_by_layout_type = defaultdict(list)
        data_by_string_length = defaultdict(list)
        data_by_layout_and_length = defaultdict(list)

        #images = np.load(IMAGE_FILE % set_name)

        with open(QUESTION_FILE % set_name) as question_f, \
             open(PARSE_FILE % set_name) as parse_f, \
             open(ANN_FILE % set_name) as ann_f:

            i = 0
            for question, parse_str, answer in zip(question_f, parse_f, ann_f):

                question = question.strip().split()
                question = ["<s>"] + question + ["</s>"]
                question = [QUESTION_INDEX.index(w) for w in question]
                parse = parse_tree(parse_str.strip())
                layout = parse_to_layout(parse, config, modules)

                answer = ANSWER_INDEX.index(answer)
                image_path = IMAGE_FILE % (set_name, i)
                raw_image_path = RAW_IMAGE_FILE % (set_name, i)
                with np.load(image_path) as zdata:
                    assert len(zdata.keys()) == 1
                    image = zdata[zdata.keys()[0]]
                datum = ShapesDatum(i, question, layout, image, answer, raw_image_path)
                datum.raw_query = parse_str
                datum.question = question

                data.add(datum)
                #data_by_layout_type[datum.layout.modules].append(datum)
                #data_by_string_length[len(datum.string)].append(datum)
                #data_by_layout_and_length[(datum.layout.modules, len(datum.string))].append(datum)

                i += 1

        self.data = data
        #self.by_layout_type = data_by_layout_type
        #self.by_string_length = data_by_string_length
        #self.by_layout_and_length = data_by_layout_and_length

        logging.info("%s:", set_name.upper())
        logging.info("%s items", len(self.data))
        logging.info("%s words", len(QUESTION_INDEX))
        logging.info("%s predicates", len(MODULE_INDEX))
        logging.info("%s answers", len(ANSWER_INDEX))
        #logging.info("%s layouts", len(self.by_layout_type.keys()))
        logging.info("")

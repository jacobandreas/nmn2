#!/usr/bin/env python2

from misc.datum import Datum, Layout
from misc.indices import QUESTION_INDEX, LAYOUT_INDEX, ANSWER_INDEX
from misc.parse import parse_tree
#from models.modules import \
#        AttAnswerModule, DetectModule, DenseAnswerModule, ConjModule, \
#        RedetectModule

from collections import defaultdict
import logging
import numpy as np

QUESTION_FILE = "data/daquar/%s/%s.questions.txt"
PARSE_FILE = "data/daquar/%s/%s.questions.sp"
ANN_FILE = "data/daquar/%s/%s.answers.txt"
IMAGE_FILE = "data/daquar/images/conv/%s.png.npz"
RAW_IMAGE_FILE = "data/daquar/images/raw/%s.png"

TRAIN_IMAGES_FILE = "data/daquar/train.txt"
VAL_IMAGES_FILE = "data/daquar/val.txt"

def parse_to_layout(parse):
    return Layotu((), ())
    return Layout(*parse_to_layout_helper(parse, internal=False))

def parse_to_layout_helper(parse, internal):
    if isinstance(parse, str):
        return (DetectModule, LAYOUT_INDEX.index(parse))
    else:
        head = parse[0]
        head_idx = LAYOUT_INDEX.index(parse)
        if internal:
            if head == "and":
                mod_head = ConjModule
            else:
                mod_head = RedetectModule
        else:
            if head == "how many":
                mod_head = DenseAnswerModule
            else:
                mod_head = AttAnswerModule

        below = [parse_to_layout_helper(child, internal=True) for child in parse[1:]]
        mods_below, indices_below = zip(*below)
        return (mod_head,) + tuple(mods_below), (head_idx,) + tuple(indices_below)

class DaquarDatum(Datum):
    def __init__(self, string, layout, image_id, answer):
        self.string = string
        self.layout = layout
        self.image_id = image_id
        self.answer = answer
        self.outputs = [answer]

        self.input_path = IMAGE_FILE % image_id
        self.image_path = RAW_IMAGE_FILE % image_id

    def load_input(self):
        with np.load(self.input_path) as zdata:
            assert len(zdata.keys()) == 1
            image_data = zdata[zdata.keys()[0]]
        return image_data

class DaquarTask:
    def __init__(self, config):
        self.train = DaquarTaskSet(config, "train", TRAIN_IMAGES_FILE)
        self.val = DaquarTaskSet(config, "train", VAL_IMAGES_FILE)
        self.test = DaquarTaskSet(config, "test")

class DaquarTaskSet:
    def __init__(self, config, set_name, filter_file=None):
        self.config = config
        size = config.train_size

        data = set()
        data_by_layout_type = defaultdict(list)
        data_by_string_length = defaultdict(list)
        data_by_layout_and_length = defaultdict(list)

        with open(QUESTION_FILE % (size, set_name)) as question_f, \
             open(PARSE_FILE % (size, set_name)) as parse_f, \
             open(ANN_FILE % (size, set_name)) as ann_f:

            img_filter = None
            if filter_file is not None:
                img_filter = set()
                with open(filter_file) as filt_h:
                    for line in filt_h:
                        img_filter.add(line.strip())

                
            for question, parse_str, answer in zip(question_f, parse_f, ann_f):
                question = question.strip()
                parse_str = parse_str.strip()
                #parse_str = "(what object)"
                answer = answer.strip()
                words = question.split()
                image_id = words[-2]
                words = ["<s>"] + words[:-4] + ["</s>"]

                # TODO multi answer
                if "," in answer:
                    continue
                if img_filter is not None and image_id not in img_filter:
                    continue

                answer = ANSWER_INDEX.index(answer)

                indexed_words = [QUESTION_INDEX.index(w) for w in words]

                parse = parse_tree(parse_str)
                #if parse[0] != "color":
                #    continue
                layout = parse_to_layout(parse)
                datum = DaquarDatum(indexed_words, layout, image_id, answer)

                data.add(datum)
                data_by_layout_type[datum.layout.modules].append(datum)
                data_by_string_length[len(datum.string)].append(datum)
                data_by_layout_and_length[(datum.layout.modules, len(datum.string))].append(datum)

        self.data = data
        self.by_layout_type = data_by_layout_type
        self.by_string_length = data_by_string_length
        self.by_layout_and_length = data_by_layout_and_length

        logging.info("%s:", set_name.upper())
        logging.info("%s items", len(self.data))
        logging.info("%s words", len(QUESTION_INDEX))
        logging.info("%s functions", len(LAYOUT_INDEX))
        logging.info("%s answers", len(ANSWER_INDEX))
        logging.info("%s layouts", len(self.by_layout_type.keys()))
        logging.info("")

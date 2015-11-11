#!/usr/bin/env python2

from datum import Datum, Layout
from indices import STRING_INDEX, LAYOUT_INDEX, ANSWER_INDEX
from parse import parse_tree
from models.modules import AttAnswerModule, DetectModule, DenseAnswerModule

from collections import defaultdict
import json
import logging
import numpy as np
import os

STRING_FILE = "data/images/Questions/OpenEnded_mscoco_%s2014_questions.json"
PARSE_FILE = "data/images/Questions/OpenEnded_mscoco_%s2014_questions.sp"
ANN_FILE = "data/images/Annotations/mscoco_%s2014_annotations.json"
IMAGE_FILE = "data/images/Images/%s2014/conv/COCO_%s2014_%012d.jpg.npz"
RAW_IMAGE_FILE = "data/images/Images/%s2014/COCO_%s2014_%012d.jpg"

LEGAL_QUERIES = set([
    #"count", 
    "color",
    "is_there"
])

def parse_to_layout(parse):

    #if parse not in [("color", "cat"), ("color", "shirt")]:
    #    return None

    #if not isinstance(parse, tuple):
    #    return None
    #if parse[0] not in LEGAL_QUERIES:
    #    return None
    #if isinstance(parse[1], tuple):
    #    return None

    layout_modules = [None, None]
    layout_indices = [None, None]

    if parse[0] in ("is", "is_there", "count"):
        layout_modules[0] = DenseAnswerModule
    else:
        layout_modules[0] = AttAnswerModule
    #else:
    #    print parse
    #    exit()
    #elif parse[0] == "count":
    #    layout_modules[0] = DenseAnswerModule
    layout_indices[0] = LAYOUT_INDEX.index(parse[0])

    layout_modules[1] = DetectModule
    layout_indices[1] = LAYOUT_INDEX.index(parse[1])

    layout = Layout(tuple(layout_modules), tuple(layout_indices))
    return layout

class ImageDatum(Datum):
    def __init__(self, id, string, layout, input_set, input_id, outputs):
        Datum.__init__(self)
        self.id = id
        self.string = string
        self.layout = layout
        self.input_set = input_set
        self.input_id = input_id
        self.outputs = outputs

        self.input_path = IMAGE_FILE % (self.input_set, self.input_set, self.input_id)
        self.image_path = RAW_IMAGE_FILE % (self.input_set, self.input_set, self.input_id)

        if not os.path.exists(self.input_path):
            raise IOError("No such processed image: " + self.input_path)
        if not os.path.exists(self.input_path):
            raise IOError("No such source image: " + self.image_paht)

    def load_input(self):
        with np.load(self.input_path) as zdata:
            assert len(zdata.keys()) == 1
            image_data = zdata[zdata.keys()[0]]
        return image_data
        #channels, width, height = image_data.shape
        #padded = np.zeros((channels, self.pad_to_width, self.pad_to_height))
        #padded[:,:width,:height] = image_data
        #return padded

class ImageTask:

    def __init__(self, config):
        train_size = \
            {
                "tiny":  10,
                "small": 1000,
                "med":   10000,
                "large": None
            }[config.train_size]

        other_size = None if config.train_size == "large" else train_size

        self.train = ImageTaskSet(config, "train", train_size)
        self.val = ImageTaskSet(config, "val", other_size)
        self.test = ImageTaskSet(config, "test", other_size)


class ImageTaskSet:

    def __init__(self, config, set_name, size=None):
        self.config = config
        self.size = size

        data_by_id = dict()
        data_by_layout_type = defaultdict(list)

        if set_name == "test":
            self.by_id = data_by_id
            self.by_layout_type = data_by_layout_type
            self.layout_types = set()
            return

        with open(STRING_FILE % set_name) as question_f, \
             open(PARSE_FILE % set_name) as parse_f:
            questions = json.load(question_f)["questions"]
            parses = [l.strip() for l in parse_f]
            assert len(questions) == len(parses)
            pairs = zip(questions, parses)
            if size is not None:
                pairs = pairs[:size]
            for question, parse_str in pairs:
                id = question["question_id"]
                indexed_question = \
                    [STRING_INDEX.index(w) for w in question["question"].split()]
                parse = parse_tree(parse_str)
                layout = parse_to_layout(parse)
                if layout is None:
                    continue
                image_id = question["image_id"]
                #if os.path.exists(IMAGE_FILE % (set_name, set_name, image_id)):
                try:
                    datum = ImageDatum( id, indexed_question, layout, set_name, image_id, [])
                    data_by_id[id] = datum
                except IOError as e:
                    pass
                    

        with open(ANN_FILE % set_name) as ann_f:
            annotations = json.load(ann_f)["annotations"]
            for ann in annotations:
                question_id = ann["question_id"]
                if question_id not in data_by_id:
                    continue

                answer_counter = defaultdict(lambda: 0)
                for ans in ann["answers"]:
                    ans_words = ans["answer"]
                    if " " in ans_words or "/" in ans_words or "," in ans_words:
                        continue
                    answer_counter[ans_words] += 1
                    #ans_indexed = ANSWER_INDEX.index(ans_words)
                    #data_by_id[question_id].outputs.append(ans_indexed)

                counted_answers = [(count, word) for word, count in
                        answer_counter.items()]
                sorted_answers = sorted(counted_answers)
                if len(sorted_answers) == 0:
                    del data_by_id[question_id]
                    continue
                best_count = sorted_answers[-1][0]
                if best_count == 1:
                    del data_by_id[question_id]
                    continue
                best_answer = sorted_answers[-1][1]
                best_answer_indexed = ANSWER_INDEX.index(best_answer)
                data_by_id[question_id].outputs.append(best_answer_indexed)

        for datum in data_by_id.values():
            data_by_layout_type[datum.layout.modules].append(datum)

        self.by_id = data_by_id
        self.by_layout_type = data_by_layout_type
        self.layout_types = data_by_layout_type.keys()

        logging.info("%s:", set_name.upper())
        logging.info("%s items", len(self.by_id))
        logging.info("%d answers", len(ANSWER_INDEX))
        logging.info("%d layouts", len(self.layout_types))
        logging.info("")

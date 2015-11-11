#!/usr/bin/env python2

from datum import Datum, Layout
from indices import STRING_INDEX, LAYOUT_INDEX, ANSWER_INDEX, UNK
from parse import parse_tree
from models.modules import *

from collections import defaultdict
import logging
import numpy as np
import os

STRING_FILE = "data/cocoqa/%s/questions.txt"
PARSE_FILE = "data/cocoqa/%s/questions.sp"
ANN_FILE = "data/cocoqa/%s/answers.txt"
IMAGE_ID_FILE = "data/cocoqa/%s/img_ids.txt"
IMAGE_FILE = "data/vqa/Images/%s2014/conv/COCO_%s2014_%012d.jpg.npz"
RAW_IMAGE_FILE = "data/vqa/Images/%s2014/COCO_%s2014_%012d.jpg"

def parse_to_layout(parse):
    return Layout(*parse_to_layout_helper(parse, internal=False))

def parse_to_layout_helper(parse, internal):
    if isinstance(parse, str):
        #return (DetectModule, LAYOUT_INDEX.index(parse))
        return (DetectModule, LAYOUT_INDEX.get_or_else(parse, LAYOUT_INDEX[UNK]))
    else:
        head = parse[0]
        #head_idx = LAYOUT_INDEX.index(parse)
        head_idx = LAYOUT_INDEX.get_or_else(head, LAYOUT_INDEX[UNK])
        if internal:
            if head == "and":
                mod_head = ConjModule
            else:
                mod_head = RedetectModule
        else:
            if head == "count":
                mod_head = DenseAnswerModule
            #elif head == "where":
            #    mod_head = AttAnswerModuleCopy
            else:
                mod_head = AttAnswerModule

        below = [parse_to_layout_helper(child, internal=True) for child in parse[1:]]
        mods_below, indices_below = zip(*below)
        return (mod_head,) + tuple(mods_below), (head_idx,) + tuple(indices_below)

class CocoQADatum(Datum):
    def __init__(self, string, layout, image_id, answer, coco_set_name):
        self.string = string
        self.layout = layout
        self.image_id = image_id
        self.answer = answer
        self.outputs = [answer]

        self.input_path = IMAGE_FILE % (coco_set_name, coco_set_name, image_id)
        self.image_path = RAW_IMAGE_FILE % (coco_set_name, coco_set_name, image_id)

        if not os.path.exists(self.input_path):
            raise IOError("No such processed image: " + self.input_path)
        if not os.path.exists(self.input_path):
            raise IOError("No such source image: " + self.image_paht)

    def load_input(self):
        with np.load(self.input_path) as zdata:
            assert len(zdata.keys()) == 1
            image_data = zdata[zdata.keys()[0]]

        pos_data = np.zeros((4, image_data.shape[1], image_data.shape[2]))

        pos_data[0,...] = np.linspace(-1, 1, image_data.shape[1])[:,np.newaxis]
        pos_data[1,...] = np.linspace(-1, 1, image_data.shape[2])[np.newaxis,:]
        pos_data[2,...] = pos_data[0,...] ** 2
        pos_data[3,...] = pos_data[1,...] ** 2

        return np.concatenate((image_data, pos_data), axis=0)

class CocoQATask:
    def __init__(self, config):
        self.train = CocoQATaskSet(config, "train")
        self.val = CocoQATaskSet(config, "val")
        self.test = CocoQATaskSet(config, "test")


class CocoQATaskSet:
    def __init__(self, config, set_name):
        self.config = config

        data = set()
        data_by_layout_type = defaultdict(list)
        data_by_string_length = defaultdict(list)
        data_by_layout_and_length = defaultdict(list)

        if set_name == "val":
            self.data = data
            self.by_layout_type = data_by_layout_type
            self.by_string_length = data_by_string_length
            self.by_layout_and_length = data_by_layout_and_length
            return

        if set_name == "train":
            # TODO better index
            pred_counter = defaultdict(lambda: 0)
            with open(PARSE_FILE % set_name) as parse_f:
                for parse_str in parse_f:
                    parse_preds = parse_str.strip() \
                                           .replace("'", "") \
                                           .replace("(", "") \
                                           .replace(")", "") \
                                           .split()
                    for pred in parse_preds:
                        pred_counter[pred] += 1
            for pred, count in pred_counter.items():
                if count <= 5:
                    continue
                LAYOUT_INDEX.index(pred)

            #word_counter = defaultdict(lambda: 0)
            #with open(STRING_FILE % set_name) as string_f:
            #    for sentence in string_f:
            #        words = ["<s>"] + sentence.strip().split() + ["</s>"]
            #        for word in words:
            #            word_counter[word] += 1
            ##print word_counter
            #for word, count in word_counter.items():
            #    if count <= 1:
            #        continue
            #    STRING_INDEX.index(word)

        with open(STRING_FILE % set_name) as question_f, \
             open(PARSE_FILE % set_name) as parse_f, \
             open(ANN_FILE % set_name) as ann_f, \
             open(IMAGE_ID_FILE % set_name) as image_id_f:

            unked = 0
            i = 0
            for question, parse_str, answer, image_id in \
                    zip(question_f, parse_f, ann_f, image_id_f):
            
                question = question.strip()
                parse_str = parse_str.strip().replace("'", "")
                answer = answer.strip()
                image_id = int(image_id.strip())
                words = question.split()
                words = ["<s>"] + words + ["</s>"]

                parse = parse_tree(parse_str)

                answer = ANSWER_INDEX.index(answer)
                words = [STRING_INDEX.index(w) for w in words]
                #words = [STRING_INDEX.get_or_else(w, STRING_INDEX[UNK]) for w in words]
                if len(parse) == 1:
                    parse = parse + ("object",)
                if parse[0] == "what":
                    parse = ("what", "object")
                if parse[0] == "where":
                    parse = ("where", "WHERE__" + parse[1])
                layout = parse_to_layout(parse)

                #if parse not in [("color", "shirt"), ("color", "cat")]:
                #    continue

                #if i == 300:
                #    continue
                i += 1

                coco_set_name = "train" if set_name == "train" else "val"
                try:
                    datum = CocoQADatum(words, layout, image_id, answer, coco_set_name)
                    datum.raw_query = parse_str
                    data.add(datum)
                    data_by_layout_type[datum.layout.modules].append(datum)
                    data_by_string_length[len(datum.string)].append(datum)
                    data_by_layout_and_length[(datum.layout.modules, len(datum.string))].append(datum)
                except IOError as e:
                    pass

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

#!/usr/bin/env python2

from misc.datum import Datum, Layout
from misc.indices import QUESTION_INDEX, MODULE_INDEX, ANSWER_INDEX, UNK_ID
from misc.parse import parse_tree
from models.nmn import AttendModule, ClassifyModule

from collections import defaultdict
import logging
import numpy as np
import os

QUESTION_FILE = "data/cocoqa/%s/questions.txt"
PARSE_FILE = "data/cocoqa/%s/questions.sp"
ANN_FILE = "data/cocoqa/%s/answers.txt"
IMAGE_ID_FILE = "data/cocoqa/%s/img_ids.txt"
IMAGE_FILE = "data/vqa/Images/%s2014/conv/COCO_%s2014_%012d.jpg.npz"
RAW_IMAGE_FILE = "data/vqa/Images/%s2014/raw/COCO_%s2014_%012d.jpg"

def compute_normalizers(config):
    mean = np.zeros((512,))
    mmt2 = np.zeros((512,))
    count = 0
    with open(IMAGE_ID_FILE % "train") as image_id_f:
        image_ids = image_id_f.readlines()
        if hasattr(config, "debug"):
            image_ids = image_ids[:config.debug]
        for image_id in image_ids:
            image_id = int(image_id.strip())
            with np.load(IMAGE_FILE % ("train", "train", image_id)) as zdata:
                assert len(zdata.keys()) == 1
                image_data = zdata[zdata.keys()[0]]
                sq_image_data = np.square(image_data)
                mean += np.sum(image_data, axis=(1,2))
                mmt2 += np.sum(sq_image_data, axis=(1,2))
                count += image_data.shape[1] * image_data.shape[2]
        mean /= count
        mmt2 /= count
    var = mmt2 - np.square(mean)
    std = np.sqrt(var)

    return mean, std

def parse_to_layout(parse, modules):
    return Layout(*parse_to_layout_helper(parse, modules, internal=False))

def parse_to_layout_helper(parse, modules, internal):
    if isinstance(parse, str):
        return (modules["attend"], MODULE_INDEX[parse] or UNK_ID)
    else:
        head = parse[0]
        head_idx = MODULE_INDEX[head] or UNK_ID
        if internal:
            assert False
        else:
            mod_head = modules["classify"]

        below = [parse_to_layout_helper(child, modules, internal=True)
                 for child in parse[1:]]
        mods_below, indices_below = zip(*below)
        return (mod_head,) + tuple(mods_below), \
                (head_idx,) + tuple(indices_below)

class CocoQADatum(Datum):
    def __init__(self, question, layouts, image_id, answer, coco_set_name, mean,
            std):
        self.question = question
        self.layouts = layouts
        self.layout = layouts[0]
        self.image_id = image_id
        self.answers = [answer]

        self.id = image_id

        self.input_path = IMAGE_FILE % (coco_set_name, coco_set_name, image_id)
        self.image_path = RAW_IMAGE_FILE % (coco_set_name, coco_set_name,
                                            image_id)

        if not os.path.exists(self.input_path):
            raise IOError("No such processed image: " + self.input_path)
        if not os.path.exists(self.input_path):
            raise IOError("No such source image: " + self.image_paht)

        self.mean = mean[:,np.newaxis,np.newaxis]
        self.std = std[:,np.newaxis,np.newaxis]

    def load_image(self):
        with np.load(self.input_path) as zdata:
            assert len(zdata.keys()) == 1
            image_data = zdata[zdata.keys()[0]]
        image_data -= self.mean
        image_data /= self.std

        return image_data

class CocoQATask:
    def __init__(self, config):
        modules = {
            "attend": AttendModule(config.model),
            "classify": ClassifyModule(config.model)
        }

        mean, std = compute_normalizers(config.task)

        self.train = CocoQATaskSet(config.task, "train", modules, mean, std)
        self.val = CocoQATaskSet(config.task, "val", modules, mean, std)
        self.test = CocoQATaskSet(config.task, "test", modules, mean, std)
        self.val, self.test = self.test, self.val

class CocoQATaskSet:
    def __init__(self, config, set_name, modules, mean, std):
        self.config = config

        data = set()
        data_by_layout_type = defaultdict(list)
        #data_by_question_length = defaultdict(list)
        #data_by_layout_and_length = defaultdict(list)

        if set_name == "val":
            self.data = data
            self.by_layout_type = data_by_layout_type
            #self.by_question_length = data_by_question_length
            #self.by_layout_and_length = data_by_layout_and_length
            self.batch_keys = set()
            self.batches = {}
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
                MODULE_INDEX.index(pred)

            word_counter = defaultdict(lambda: 0)
            with open(QUESTION_FILE % set_name) as question_f:
                for sentence in question_f:
                    words = ["<s>"] + sentence.strip().lower().split() + ["</s>"]
                    for word in words:
                        word_counter[word] += 1
            for word, count in word_counter.items():
                if count <= 1:
                    continue
                QUESTION_INDEX.index(word)

        with open(QUESTION_FILE % set_name) as question_f, \
             open(PARSE_FILE % set_name) as parse_f, \
             open(ANN_FILE % set_name) as ann_f, \
             open(IMAGE_ID_FILE % set_name) as image_id_f:

            unked = 0
            i = 0
            for question, parse_str, answer, image_id in \
                    zip(question_f, parse_f, ann_f, image_id_f):

                question = question.strip().lower()
                parse_str = parse_str.strip().replace("'", "")
                answer = answer.strip()
                image_id = int(image_id.strip())
                words = question.split()
                #words = parse_str.replace("(", "").replace(")", "").split()
                words = ["<s>"] + words + ["</s>"]

                parse = parse_tree(parse_str)

                answer = ANSWER_INDEX.index(answer)
                words = [QUESTION_INDEX[w] or UNK_ID for w in words]

                if len(parse) == 1:
                    parse = parse + ("object",)
                #if parse[0] == "what":
                #    parse = ("what", "object")
                #if parse[0] == "where":
                #    parse = ("where", "WHERE__" + parse[1])
                layout = parse_to_layout(parse, modules)

                if hasattr(config, "debug") and i == config.debug:
                    break

                i += 1

                coco_set_name = "train" if set_name == "train" else "val"
                try:
                    datum = CocoQADatum(
                            words, [layout], image_id, answer, coco_set_name,
                            mean, std)
                    datum.raw_query = parse_str
                    data.add(datum)
                    data_by_layout_type[datum.layout.modules].append(datum)
                    #data_by_question_length[len(datum.question)].append(datum)
                    #data_by_layout_and_length[(datum.layout.modules, len(datum.question))].append(datum)
                except IOError as e:
                    pass

        self.data = data
        self.by_layout_type = data_by_layout_type
        #self.by_question_length = data_by_question_length
        #self.by_layout_and_length = data_by_layout_and_length

        logging.info("%s:", set_name.upper())
        logging.info("%s items", len(self.data))
        logging.info("%s words", len(QUESTION_INDEX))
        logging.info("%s functions", len(MODULE_INDEX))
        logging.info("%s answers", len(ANSWER_INDEX))
        #logging.info("%s layouts", len(self.by_layout_type.keys()))
        logging.info("")

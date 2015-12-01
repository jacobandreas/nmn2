#!/usr/bin/env python2

from misc.datum import Datum, Layout
from misc.indices import QUESTION_INDEX, MODULE_INDEX, ANSWER_INDEX, UNK_ID
from misc.parse import parse_tree
from models.nmn import AttendModule, ClassifyModule, MeasureModule

from collections import defaultdict
import json
import logging
import numpy as np
import os
import re

QUESTION_FILE = "data/vqa/Questions/OpenEnded_mscoco_%s_questions.json"
PARSE_FILE = "data/vqa/Questions/%s.sp"
ANN_FILE = "data/vqa/Annotations/mscoco_%s_annotations.json"
IMAGE_FILE = "data/vqa/Images/%s/conv/COCO_%s_%012d.jpg.npz"
RAW_IMAGE_FILE = "data/vqa/Images/%s/raw/COCO_%s_%012d.jpg"

MIN_COUNT = 10

def proc_question(question):
    qstr = question.lower().strip()
    if qstr[-1] == "?":
        qstr = qstr[:-1]
    words = qstr.split()
    words = ["<s>"] + words + ["</s>"]
    return words

def prepare_indices():
    set_name = "train2014"

    word_counts = defaultdict(lambda: 0)
    with open(QUESTION_FILE % set_name) as question_f:
        questions = json.load(question_f)["questions"]
        for question in questions:
            words = proc_question(question["question"])
            for word in words:
                word_counts[word] += 1
    for word, count in word_counts.items():
        if count >= MIN_COUNT:
            QUESTION_INDEX.index(word)
    
    pred_counts = defaultdict(lambda: 0)
    with open(PARSE_FILE % set_name) as parse_f:
        for line in parse_f:
            parts = line.strip().replace("(", "").replace(")", "").split()
            for part in parts:
                pred_counts[part] += 1
    for pred, count in pred_counts.items():
        if count >= MIN_COUNT:
            MODULE_INDEX.index(pred)

    answer_counts = defaultdict(lambda: 0)
    with open(ANN_FILE % set_name) as ann_f: 
        annotations = json.load(ann_f)["annotations"]
        for ann in annotations:
            for answer in ann["answers"]:
                if answer["answer_confidence"] != "yes":
                    continue
                word = answer["answer"]
                if re.search(r"[^\w\s]", word):
                    continue
                answer_counts[word] += 1

    keep_answers = reversed(sorted([(c, a) for a, c in answer_counts.items()]))
    keep_answers = list(keep_answers)[:1000]
    for count, answer in keep_answers:
        ANSWER_INDEX.index(answer)

def compute_normalizers(config):
    mean = np.zeros((512,))
    mmt2 = np.zeros((512,))
    count = 0
    with open(QUESTION_FILE % "train2014") as question_f:
        questions = json.load(question_f)["questions"]
        image_ids = [q["image_id"] for q in questions]
        if hasattr(config, "debug"):
            image_ids = image_ids[:config.debug]
        for image_id in image_ids:
            with np.load(IMAGE_FILE % ("train2014", "train2014", image_id)) as zdata:
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
    layout_modules = [None, None]
    layout_indices = [None, None]

    if parse[0] in ("is1", "is2"):
        layout_modules[0] = modules["measure"]
    else:
        layout_modules[0] = modules["classify"]
    layout_indices[0] = MODULE_INDEX[parse[0]] or UNK_ID

    layout_modules[1] = modules["attend"]
    layout_indices[1] = MODULE_INDEX[parse[1]] or UNK_ID

    layout = Layout(tuple(layout_modules), tuple(layout_indices))
    return layout

class VqaDatum(Datum):
    def __init__(self, id, question, layout, input_set, input_id, answers, mean, std):
        Datum.__init__(self)
        self.id = id
        self.question = question
        self.layout = layout
        self.input_set = input_set
        self.input_id = input_id
        self.answers = answers

        self.input_path = IMAGE_FILE % (self.input_set, self.input_set, self.input_id)
        self.image_path = RAW_IMAGE_FILE % (self.input_set, self.input_set, self.input_id)

        self.mean = mean[:,np.newaxis,np.newaxis]
        self.std = std[:,np.newaxis,np.newaxis]

        if not os.path.exists(self.input_path):
            raise IOError("No such processed image: " + self.input_path)
        if not os.path.exists(self.input_path):
            raise IOError("No such source image: " + self.image_paht)

    def load_image(self):
        with np.load(self.input_path) as zdata:
            assert len(zdata.keys()) == 1
            image_data = zdata[zdata.keys()[0]]
        image_data -= self.mean
        image_data /= self.std
        return image_data

class VqaTask:

    def __init__(self, config):
        prepare_indices()
        logging.debug("prepared indices")

        # modules need access to indices to compute some layer sizes
        modules = {
            "attend": AttendModule(config.model),
            "classify": ClassifyModule(config.model),
            "measure": MeasureModule(config.model)
        }

        mean, std = compute_normalizers(config.task)
        logging.debug("computed image feature normalizers")

        self.train = VqaTaskSet(config.task, ["train2014", "val2014"], modules, mean, std)
        self.val = VqaTaskSet(config.task, ["test-dev2015"], modules, mean, std)
        self.test = VqaTaskSet(config.task, [], modules, mean, std)
        #self.train = VqaTaskSet(config.task, ["train2014"], modules, mean, std)
        #self.val = VqaTaskSet(config.task, ["val2014"], modules, mean, std)
        #self.test = VqaTaskSet(config.task, ["test2015"], modules, mean, std)


class VqaTaskSet:

    def __init__(self, config, set_names, modules, mean, std):
        size = config.debug if hasattr(config, "debug") else None

        self.by_id = dict()
        self.by_layout_type = defaultdict(list)

        for set_name in set_names:
            self.load_set(config, set_name, size, modules, mean, std)

        for datum in self.by_id.values():
            self.by_layout_type[datum.layout.modules].append(datum)

        self.layout_types = self.by_layout_type.keys()

        logging.info("%s:", ", ".join(set_names).upper())
        logging.info("%s items", len(self.by_id))
        logging.info("%d answers", len(ANSWER_INDEX))
        logging.info("%d predicates", len(MODULE_INDEX))
        logging.info("%d words", len(QUESTION_INDEX))
        logging.info("%d layouts", len(self.layout_types))
        logging.info("")

    def load_set(self, config, set_name, size, modules, mean, std):
        with open(QUESTION_FILE % set_name) as question_f, \
             open(PARSE_FILE % set_name) as parse_f:
            questions = json.load(question_f)["questions"]
            parses = [l.strip() for l in parse_f]
            assert len(questions) == len(parses)
            pairs = zip(questions, parses)
            if size is not None:
                pairs = pairs[:size]
            for question, parse_str in pairs:
                id = question["question_id"]
                question_str = proc_question(question["question"])
                indexed_question = \
                    [QUESTION_INDEX[w] or UNK_ID for w in question_str]
                
                parse = parse_tree(parse_str)
                layout = parse_to_layout(parse, modules)
                if layout is None:
                    continue
                image_id = question["image_id"]
                try:
                    image_set_name = "test2015" if set_name == "test-dev2015" else set_name
                    datum = VqaDatum(id, indexed_question, layout, image_set_name, image_id, [], mean, std)
                    datum.raw_query = parse_str
                    self.by_id[id] = datum
                except IOError as e:
                    print e
                    pass
                    

        if set_name not in ("test2015", "test-dev2015"):
            with open(ANN_FILE % set_name) as ann_f:
                annotations = json.load(ann_f)["annotations"]
                for ann in annotations:
                    question_id = ann["question_id"]
                    if question_id not in self.by_id:
                        continue

                    answer_counter = defaultdict(lambda: 0)
                    answers = [a["answer"] for a in ann["answers"]]
                    indexed_answers = [ANSWER_INDEX[a] or UNK_ID for a in answers]
                    self.by_id[question_id].answers = indexed_answers

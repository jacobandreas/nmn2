#!/usr/bin/env python2

from misc.datum import Datum, Layout
from misc.indices import QUESTION_INDEX, MODULE_INDEX, ANSWER_INDEX, UNK_ID
from misc.parse import parse_tree
from models.nmn import MLPFindModule, DescribeModule, ExistsModule, AndModule

from collections import defaultdict
import json
import logging
import numpy as np
import os
import re

QUESTION_FILE = "data/vqa/Questions/OpenEnded_mscoco_%s_questions.json"
SINGLE_PARSE_FILE = "data/vqa/Questions/%s.sp"
MULTI_PARSE_FILE = "data/vqa/Questions/%s.sps2"
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

def prepare_indices(config):
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
    with open(MULTI_PARSE_FILE % set_name) as parse_f:
        for line in parse_f:
            parts = line.strip().replace("(", "").replace(")", "").replace(";", " ").split()
            for part in parts:
                pred_counts[part] += 1
    for pred, count in pred_counts.items():
        if count >= 10 * MIN_COUNT:
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
    keep_answers = list(keep_answers)[:config.answers]
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

def parse_to_layout_helper(parse, config, modules):
    if isinstance(parse, str):
        return modules["find"], MODULE_INDEX[parse] or UNK_ID
    head = parse[0]
    below = [parse_to_layout_helper(c, config, modules) for c in parse[1:]]
    modules_below, labels_below = zip(*below)
    modules_below = tuple(modules_below)
    labels_below = tuple(labels_below)
    if head == "and":
        module_head = modules["and"]
    else:
        module_head = modules["describe"]
    label_head = MODULE_INDEX[head] or UNK_ID
    modules_here = (module_head,) + modules_below
    labels_here = (label_head,) + labels_below
    return modules_here, labels_here

def parse_to_layout(parse, config, modules):
    modules, indices = parse_to_layout_helper(parse, config, modules)
    return Layout(modules, indices)

class VqaDatum(Datum):
    def __init__(self, id, question, parses, layouts, input_set, input_id, answers, mean, std):
        Datum.__init__(self)
        self.id = id
        self.question = question
        self.parses = parses
        self.layouts = layouts
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

    def load_features(self):
        with np.load(self.input_path) as zdata:
            assert len(zdata.keys()) == 1
            image_data = zdata[zdata.keys()[0]]
        image_data -= self.mean
        image_data /= self.std
        channels, width, height = image_data.shape
        image_data = image_data.reshape((channels, width * height, 1))
        return image_data

    def load_rel_features(self):
        return None

class VqaTask:
    def __init__(self, config):
        prepare_indices(config.task)
        logging.debug("prepared indices")

        modules = {
            "find": MLPFindModule(config.model),
            "describe": DescribeModule(config.model),
            "exists": ExistsModule(config.model),
            "and": AndModule(config.model),
        }

        mean, std = compute_normalizers(config.task)
        logging.debug("computed image feature normalizers")
        logging.debug("using %s chooser", config.task.chooser)

        self.train = VqaTaskSet(config.task, ["train2014", "val2014"], modules, mean, std)
        self.val = VqaTaskSet(config.task, ["test-dev2015"], modules, mean, std)
        self.test = VqaTaskSet(config.task, ["test2015"], modules, mean, std)

class VqaTaskSet:
    def __init__(self, config, set_names, modules, mean, std):
        size = config.debug if hasattr(config, "debug") else None

        self.by_id = dict()
        self.by_layout_type = defaultdict(list)

        for set_name in set_names:
            self.load_set(config, set_name, size, modules, mean, std)

        for datum in self.by_id.values():
            self.by_layout_type[datum.layouts[0].modules].append(datum)
            datum.layout = datum.layouts[0]

        self.layout_types = self.by_layout_type.keys()
        self.data = self.by_id.values()

        logging.info("%s:", ", ".join(set_names).upper())
        logging.info("%s items", len(self.by_id))
        logging.info("%d answers", len(ANSWER_INDEX))
        logging.info("%d predicates", len(MODULE_INDEX))
        logging.info("%d words", len(QUESTION_INDEX))
        #logging.info("%d layouts", len(self.layout_types))
        logging.info("")

    def load_set(self, config, set_name, size, modules, mean, std):
        parse_file = MULTI_PARSE_FILE
        with open(QUESTION_FILE % set_name) as question_f, \
             open(parse_file % set_name) as parse_f:
            questions = json.load(question_f)["questions"]
            parse_groups = [l.strip() for l in parse_f]
            assert len(questions) == len(parse_groups)
            pairs = zip(questions, parse_groups)
            if size is not None:
                pairs = pairs[:size]
            for question, parse_group in pairs:
                id = question["question_id"]
                question_str = proc_question(question["question"])
                indexed_question = \
                    [QUESTION_INDEX[w] or UNK_ID for w in question_str]

                parse_strs = parse_group.split(";")
                parses = [parse_tree(p) for p in parse_strs]
                parses = [("_what", "_thing") if p == "none" else p for p in parses]
                if config.chooser == "null":
                    parses = [("_what", "_thing")]
                elif config.chooser == "cvpr":
                    if parses[0][0] == "is":
                        parses = parses[-1:]
                    else:
                        parses = parses[:1]
                elif config.chooser == "naacl":
                    pass
                else:
                    assert False

                layouts = [parse_to_layout(p, config, modules) for p in parses]
                image_id = question["image_id"]
                try:
                    image_set_name = "test2015" if set_name == "test-dev2015" else set_name
                    datum = VqaDatum(id, indexed_question, parses, layouts, image_set_name, image_id, [], mean, std)
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

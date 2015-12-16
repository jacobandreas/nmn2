#!/usr/bin/env python2

from misc.datum import Datum, Layout
from misc.indices import QUESTION_INDEX, MODULE_INDEX, ANSWER_INDEX, UNK_ID
from misc.parse import parse_tree
from models.nmn import AttendModule, ClassifyModule, MeasureModule, \
        CombineModule, ReAttendModule, LookupModule, AnswerTranslationModule

from collections import defaultdict, namedtuple
import logging
import numpy as np
import os
import re

DATA_FILE = "data/geo/environments/%s/training.txt"
PARSE_FILE = "data/geo/environments/%s/training.sps"
WORLD_FILE = "data/geo/environments/%s/world.txt"
LOCATION_FILE = "data/geo/environments/%s/locations.txt"

ENVIRONMENTS = ["fl", "ga", "mi", "nc", "ok", "pa", "sc", "tn", "va", "wv"]
#ENVIRONMENTS = ["fl"]

CATS = ["city", "state", "park", "island", "beach", "ocean", "lake", "forest",
        "major", "peninsula", "capital", "water"]

DATABASE_SIZE=10

TRAIN = 0
VAL = 1
TEST = 2

YES = "yes"
NO = "no"

World = namedtuple("World", ("name", "categories", "entities", "features"))

def parse_to_layout_helper(parse, world, config, modules):
    if isinstance(parse, str):
        if parse in world.entities:
            return modules["lookup"], world.entities[parse]
        elif parse in world.categories:
            return modules["attend"], world.categories[parse]
        else:
            logging.warn("weird predicate: %s", parse)
            return modules["attend"], UNK_ID
        #return modules["attend"], MODULE_INDEX[parse] or UNK_ID
    head = parse[0]
    below = [parse_to_layout_helper(c, world, config, modules) for c in parse[1:]]
    modules_below, labels_below = zip(*below)
    modules_below = tuple(modules_below)
    labels_below = tuple(labels_below)
    if head == "and":
        module_head = modules["combine"]
    elif head == "exists":
        module_head = modules["measure"]
    else:
        module_head = modules["re-attend"]
    label_head = MODULE_INDEX[head] or UNK_ID
    modules_here = (module_head,) + modules_below
    labels_here = (label_head,) + labels_below
    return modules_here, labels_here

def parse_to_layout(parse, world, config, modules):
    mods, indices = parse_to_layout_helper(parse, world, config, modules)

    head = mods[0] if isinstance(mods, tuple) else mods
    if not isinstance(head, MeasureModule):
        # wrap in translation module
        # TODO bad naming
        mapping = {i: ANSWER_INDEX[cat] for i, cat in enumerate(world.entities)}
        index = modules["translate"].register(world.name, mapping)
        mods = (modules["translate"], mods)
        indices = (index, indices)
    return Layout(mods, indices)

class GeoDatum(Datum):
    def __init__(self, id, question, layouts, answer, world):
        Datum.__init__(self)
        self.id = id
        self.question = question
        self.layouts = layouts
        self.answers = [answer]
        self.world = world

    def load_image(self):
        return self.world.features

class GeoTask:
    def __init__(self, config):
        for cat in CATS:
            MODULE_INDEX.index(cat)
        modules = {
            "attend": AttendModule(config.model),
            "lookup": LookupModule(config.model),
            "measure": MeasureModule(config.model),
            "combine": CombineModule(config.model),
            "re-attend": ReAttendModule(config.model),
            "translate": AnswerTranslationModule(config.model)
        }
        self.train = GeoTaskSet(config.task, TRAIN, modules)
        self.val = GeoTaskSet(config.task, VAL, modules)
        self.test = GeoTaskSet(config.task, TEST, modules)

class GeoTaskSet:
    def __init__(self, config, set_name, modules):
        if set_name == VAL:
            self.data = []
            return

        questions = []
        answers = []
        parse_lists = []
        worlds = []

        if config.quant:
            ANSWER_INDEX.index(YES)
            ANSWER_INDEX.index(NO)

        for i_env, environment in enumerate(ENVIRONMENTS):
            if i_env == config.fold and set_name == TRAIN:
                continue
            if i_env != config.fold and set_name == TEST:
                continue

            #database = np.random.random((101, 102, 1))

            locations = dict()
            with open(LOCATION_FILE % environment) as loc_f:
                for line in loc_f:
                    parts = line.strip().split(";")
                    locations[parts[0]] = -0.5 * np.ones(len(CATS))

            with open(WORLD_FILE % environment) as world_f:
                for line in world_f:
                    parts = line.strip().split(";")
                    cat = parts[0][1:]
                    if cat not in CATS:
                        continue
                    places = parts[1].split(",")
                    cat_id = CATS.index(cat)
                    for place in places:
                        locations[place][cat_id] = 0.5

            loc_keys = sorted(locations.keys())
            clean_loc_keys = [k.lower().replace(" ", "_") for k in loc_keys]

            for k in clean_loc_keys:
                ANSWER_INDEX.index(k)

            cat_index = {cat: i for i, cat in enumerate(CATS)}
            loc_index = {loc: i for i, loc in enumerate(clean_loc_keys)}
            database = np.asarray([locations[l] for l in loc_keys])
            database = database.T
            database = database.reshape((len(CATS), len(loc_keys), 1))

            features = np.zeros((database.shape[0], DATABASE_SIZE, 1))
            features[:, :database.shape[1], :] = database

            world = World(environment, cat_index, loc_index, features)

            with open(DATA_FILE % environment) as data_f:
                for line in data_f:
                    line = line.strip()
                    if line == "" or line[0] == "#":
                        continue

                    parts = line.split(";")

                    question = parts[0]
                    if question[-1] != "?":
                        question += " ?"
                    question = question.lower()
                    questions.append(question)

                    answer = parts[1].lower().replace(" ", "_")
                    if config.quant and question[:2] in ("is", "are"):
                        answer = YES if answer else NO
                    answers.append(answer)

                    worlds.append(world)

            with open(PARSE_FILE % environment) as parse_f:
                for line in parse_f:
                    parse_strs = line.strip().split(";")
                    trees = [parse_tree(s) for s in parse_strs]
                    if not config.quant:
                        trees = [t for t in trees if t[0] != "exists"]
                    parse_lists.append(trees)

        assert len(questions) == len(parse_lists)

        data = []
        i_datum = 0
        for question, answer, parse_list, world in \
                zip(questions, answers, parse_lists, worlds):
            tokens = ["<s>"] + question.split() + ["</s>"]

            # TODO
            parse_list = parse_list[-1:]

            indexed_question = [QUESTION_INDEX.index(w) for w in tokens]
            indexed_answer = \
                    tuple(ANSWER_INDEX[a] for a in answer.split(",") if a != "")
            assert all(a is not None for a in indexed_answer)
            layouts = [parse_to_layout(p, world, config, modules) for p in parse_list]

            data.append(GeoDatum(
                    i_datum, indexed_question, layouts, indexed_answer, world))
            i_datum += 1

        self.data = data
        #self.data = data[:10]

        logging.info("%s:", set_name)
        logging.info("%s items", len(self.data))
        logging.info("%s words", len(QUESTION_INDEX))
        logging.info("%s functions", len(MODULE_INDEX))
        logging.info("%s answers", len(ANSWER_INDEX))

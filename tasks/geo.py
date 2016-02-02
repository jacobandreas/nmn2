#!/usr/bin/env python2

from misc.datum import Datum, Layout
from misc.indices import QUESTION_INDEX, MODULE_INDEX, ANSWER_INDEX, UNK_ID
from misc.parse import parse_tree
import misc.util
from models.nmn import MultiplicativeFindModule, LookupModule, AndModule, \
        ExistsModule, RelateModule, AnswerAdaptor

from collections import defaultdict, namedtuple
import logging
import numpy as np
import os
import re
import xml.etree.ElementTree as ET

DATA_FILE = "data/geo/environments/%s/training.txt"
PARSE_FILE = "data/geo/environments/%s/training.sps"
WORLD_FILE = "data/geo/environments/%s/world.txt"
LOCATION_FILE = "data/geo/environments/%s/locations.txt"

ENVIRONMENTS = ["fl", "ga", "mi", "nc", "ok", "pa", "sc", "tn", "va", "wv"]

CATS = ["city", "state", "park", "island", "beach", "ocean", "lake", "forest",
        "major", "peninsula", "capital"]
RELS = ["in-rel", "north-rel", "south-rel", "east-rel", "west-rel", "border-rel"]

DATABASE_SIZE=10

TRAIN = 0
VAL = 1
TEST = 2

YES = "yes"
NO = "no"

World = namedtuple("World", ("name", "entities", "entity_features", "relation_features"))

def parse_to_layout_helper(parse, world, config, modules):
    if isinstance(parse, str):
        if parse in world.entities:
            return modules["lookup"], world.entities[parse]
        else:
            return modules["find"], MODULE_INDEX.index(parse)
    head = parse[0]
    below = [parse_to_layout_helper(c, world, config, modules) for c in parse[1:]]
    modules_below, labels_below = zip(*below)
    modules_below = tuple(modules_below)
    labels_below = tuple(labels_below)
    if head == "and":
        module_head = modules["and"]
    elif head == "exists":
        module_head = modules["exists"]
    else:
        module_head = modules["relate"]
    label_head = MODULE_INDEX.index(head)
    modules_here = (module_head,) + modules_below
    labels_here = (label_head,) + labels_below
    return modules_here, labels_here

def parse_to_layout(parse, world, config, modules):
    mods, indices = parse_to_layout_helper(parse, world, config, modules)

    head = mods[0] if isinstance(mods, tuple) else mods
    if not isinstance(head, ExistsModule):
        # wrap in translation module
        # TODO bad naming
        mapping = {i: ANSWER_INDEX[ent] for ent, i in world.entities.items()}
        index = modules["answer_adaptor"].register(world.name, mapping)
        mods = (modules["answer_adaptor"], mods)
        indices = (index, indices)
    return Layout(mods, indices)

class GeoDatum(Datum):
    def __init__(self, id, question, parses, layouts, answer, world):
        Datum.__init__(self)
        self.id = id
        self.question = question
        self.parses = parses
        self.layouts = layouts
        self.answers = [answer]
        self.world = world

    def load_features(self):
        return self.world.entity_features

    def load_rel_features(self):
        return self.world.relation_features

class GeoTask:
    def __init__(self, config):
        modules = {
            "find": MultiplicativeFindModule(config.model),
            "lookup": LookupModule(config.model),
            "exists": ExistsModule(config.model),
            "and": AndModule(config.model),
            "relate": RelateModule(config.model),
            "answer_adaptor": AnswerAdaptor(config.model)
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

            places = list()
            with open(LOCATION_FILE % environment) as loc_f:
                for line in loc_f:
                    parts = line.strip().split(";")
                    places.append(parts[0])

            cats = {place: np.zeros((len(CATS),)) for place in places}
            rels = {(pl1, pl2): np.zeros((len(RELS),)) for pl1 in places for pl2 in places}

            with open(WORLD_FILE % environment) as world_f:
                for line in world_f:
                    parts = line.strip().split(";")
                    if len(parts) < 2:
                        continue
                    name = parts[0][1:]
                    places_here = parts[1].split(",")
                    if name in CATS:
                        cat_id = CATS.index(name)
                        for place in places_here:
                            cats[place][cat_id] = 1
                    elif name in RELS:
                        rel_id = RELS.index(name)
                        for place_pair in places_here:
                            pl1, pl2 = place_pair.split("#")
                            rels[pl1, pl2][rel_id] = 1
                            rels[pl2, pl1][rel_id] = -1

            clean_places = [p.lower().replace(" ", "_") for p in places]
            place_index = {place: i for (i, place) in enumerate(places)}
            clean_place_index = {place: i for (i, place) in enumerate(clean_places)}
            
            cat_features = np.zeros((len(CATS), DATABASE_SIZE, 1))
            rel_features = np.zeros((len(RELS), DATABASE_SIZE, DATABASE_SIZE))

            for p1, i_p1 in place_index.items():
                cat_features[:, i_p1, 0] = cats[p1]
                for p2, i_p2 in place_index.items():
                    rel_features[:, i_p1, i_p2] = rels[p1, p2]

            world = World(environment, clean_place_index, cat_features, rel_features)

            for place in clean_places:
                ANSWER_INDEX.index(place)

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
                    if config.quant and question.split()[0] in ("is", "are"):
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

            parse_list = parse_list[-config.k_best_parses:]

            indexed_question = [QUESTION_INDEX.index(w) for w in tokens]
            indexed_answer = \
                    tuple(ANSWER_INDEX[a] for a in answer.split(",") if a != "")
            assert all(a is not None for a in indexed_answer)
            layouts = [parse_to_layout(p, world, config, modules) for p in parse_list]

            data.append(GeoDatum(
                    i_datum, indexed_question, parse_list, layouts, indexed_answer, world))
            i_datum += 1

        self.data = data

        logging.info("%s:", set_name)
        logging.info("%s items", len(self.data))
        logging.info("%s words", len(QUESTION_INDEX))
        logging.info("%s functions", len(MODULE_INDEX))
        logging.info("%s answers", len(ANSWER_INDEX))

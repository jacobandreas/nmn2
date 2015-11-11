#!/usr/bin/env python2

from images import ImageTask
from daquar import DaquarTask
from cocoqa import CocoQATask
from shapes import ShapesTask

def load_task(config):
    if config.name == "images":
        return ImageTask(config)
    elif config.name == "daquar":
        return DaquarTask(config)
    elif config.name == "cocoqa":
        return CocoQATask(config)
    elif config.name == "shapes":
        return ShapesTask(config)
    else:
        raise NotImplementedError(
                "Don't know how to build a %s task" % config.name)

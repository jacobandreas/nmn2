#!/usr/bin/env python2

def load_task(config):
    if config.task.name == "vqa":
        from vqa import VqaTask
        return VqaTask(config)
    elif config.task.name == "cocoqa":
        from cocoqa import CocoQATask
        return CocoQATask(config)
    elif config.task.name == "shapes":
        from shapes import ShapesTask
        return ShapesTask(config)
    elif config.task.name == "geo":
        from geo import GeoTask
        return GeoTask(config)
    #elif config.task.name == "daquar":
    #    from daquar import DaquarTask
    #    return DaquarTask(config)
    else:
        raise NotImplementedError(
                "Don't know how to build a %s task" % config.task.name)

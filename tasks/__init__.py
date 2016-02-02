#!/usr/bin/env python2

def load_task(config):
    if config.task.name == "vqa":
        from vqa import VqaTask
        return VqaTask(config)
    if config.task.name == "geo":
        from geo import GeoTask
        return GeoTask(config)
    else:
        raise NotImplementedError(
                "Don't know how to build a %s task" % config.task.name)

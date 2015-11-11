#!/usr/bin/env python2

from misc import util
import models
import tasks

import argparse
import logging.config
import yaml


def main():
    config = configure()
    task = tasks.load_task(config.task)
    model = models.build_model(config.model, config.opt)

def configure():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
            "-c", "--config", dest="config", required=True, 
            help="model configuration file")
    arg_parser.add_argument(
            "-l", "--log-config", dest="log_config", default="config/log.yml", 
            help="log configuration file")

    args = arg_parser.parse_args()
    config_name = args.config.split("/")[-1].split(".")[0]

    with open(args.log_config) as log_config_f:
        log_filename = "logs/%s.log" % config_name
        log_config = yaml.load(log_config_f)
        log_config["handlers"]["fileHandler"]["filename"] = log_filename
        logging.config.dictConfig(log_config)

    with open(args.config) as config_f:
        config = util.Struct(**yaml.load(config_f))

    return config


if __name__ == "__main__":
    main()

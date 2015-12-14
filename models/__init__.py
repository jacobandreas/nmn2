#!/usr/bin/env python2

import apollocaffe
import caffe


apollocaffe.set_device(0)
#apollocaffe.set_random_seed(0)
#apollocaffe.set_cpp_loglevel(1)

def build_model(config, opt_config):
    #elif config.name == "monolithic":
    #    return MonolithicNMNModel(config, opt_config)
    #elif config.name == "lstm":
    #    return LSTMModel(config, opt_config)
    #elif config.name == "ensemble":
    #    return EnsembleModel(config, opt_config)
    #else:
    if config.name == "nmn":
        from nmn import NmnModel
        return NmnModel(config, opt_config)
    if config.name == "att":
        from att import AttModel
        return AttModel(config, opt_config)
    if config.name == "sp":
        from sp import StoredProgramModel
        return StoredProgramModel(config, opt_config)
    else:
        raise NotImplementedError(
                "Don't know how to build a %s model" % config.name)

#!/usr/bin/env python2

import apollocaffe
import caffe

#apollocaffe.set_device(0)
apollocaffe.set_random_seed(0)
#apollocaffe.set_cpp_loglevel(1)

def build_model(config, opt_config):
    #if config.name == "nmn":
    #    return NMNModel(config, opt_config)
    #elif config.name == "monolithic":
    #    return MonolithicNMNModel(config, opt_config)
    #elif config.name == "lstm":
    #    return LSTMModel(config, opt_config)
    #elif config.name == "ensemble":
    #    return EnsembleModel(config, opt_config)
    #else:
    if False:
        pass
    else:
        raise NotImplementedError(
                "Don't know how to build a %s model" % config.name)

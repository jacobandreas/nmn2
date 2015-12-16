#!/usr/bin/env python2

from collections import namedtuple

class Layout:
    def __init__(self, modules, labels):
        #assert isinstance(modules, tuple)
        #assert isinstance(labels, tuple)
        self.modules = modules
        self.labels = labels

    def __eq__(self, other):
        return isinstance(other, Layout) and \
                other.modules == self.modules and \
                other.labels == self.labels

    def __hash__(self):
        return hash(self.modules) + 3 * hash(self.labels)

    def __str__(self):
        return self.__str_helper(self.modules, self.labels)

    def __str_helper(self, modules, labels):
        if isinstance(modules, tuple):
            mhead, mtail = modules[0], modules[1:]
            ihead, itail = labels[0], labels[1:]
            mod_name = str(mhead) # mhead.__name__
            below = [self.__str_helper(m, i) for m, i in zip(mtail, itail)]
            return "(%s[%s] %s)" % (mod_name, ihead, " ".join(below))

        mod_name = str(modules)
        return "%s[%s]" % (mod_name, labels)

class Datum:
    def __init__(self):
        self.id = None
        self.string = None
        self.outputs = None

    def load_input(self):
        raise NotImplementedError()

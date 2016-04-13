#!/usr/bin/env python2

import sys

for line in sys.stdin:
    line = line.replace("(", "").replace(")", "")
    words = line.split()
    en_words = []
    for i, word in enumerate(words):
        if word in ("red", "green", "blue"):
            if i == 2:
                en_words.append(word)
            else:
                en_words.append("a " + word + " shape")
        elif word in ("circle", "square", "triangle"):
            en_words.append("a " + word)
        else:
            en_words.append(word.replace("_", " "))
    print " ".join(en_words)

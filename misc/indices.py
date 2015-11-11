#!/usr/bin/env python2

from util import Index

UNK = "*unknown*"
NULL = "*null*"

STRING_INDEX = Index()
LAYOUT_INDEX = Index()
ANSWER_INDEX = Index()

STRING_INDEX.index(UNK)
LAYOUT_INDEX.index(UNK)
ANSWER_INDEX.index(UNK)

STRING_INDEX.index(NULL)
LAYOUT_INDEX.index(NULL)
ANSWER_INDEX.index(NULL)

#!/usr/bin/env python2

from util import Index

UNK = "*unknown*"
NULL = "*null*"

QUESTION_INDEX = Index()
MODULE_INDEX = Index()
MODULE_TYPE_INDEX = Index()
ANSWER_INDEX = Index()

UNK_ID = QUESTION_INDEX.index(UNK)
MODULE_INDEX.index(UNK)
ANSWER_INDEX.index(UNK)

NULL_ID = QUESTION_INDEX.index(NULL)
#MODULE_INDEX.index(NULL)
#ANSWER_INDEX.index(NULL)

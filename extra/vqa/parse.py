#!/usr/bin/env python2

from collections import namedtuple
import itertools
import re
import sys

Node = namedtuple("Node", ["word", "tag", "parent", "rel", "path"])

BE_FORMS="is|are|was|were|has|have|had|does|do|did|be"

WH_RES = [
    r"^what (\w+) (is|are)",
    r"^what (is|are) the (\w+) of",
    r"^what (\w+) of",
    r"^(what|which|where)",
    r"(%s)" % BE_FORMS,
    r"^(how many)",
    r"^(can|could)"
]

EDGE_RE = re.compile(r"([^()]+)\((.+)-(\d+), (.+)-(\d+)\)")
CONTENT_RE = re.compile(r"NN*|VB*|JJ*")
#CONTENT_RE = re.compile(r"NN|VB|JJ")

REL_PRECEDENCE = ["root", "nsubj", "dobj", "nsubjpass", "dep", "xcomp"]

def precedence(rel):
    if "nmod:" in rel:
        return len(REL_PRECEDENCE)
    if "conj:" in rel:
        return len(REL_PRECEDENCE) + 1
    if "acl:" in rel:
        return len(REL_PRECEDENCE) + 1
    return REL_PRECEDENCE.index(rel)

class LfParser(object):
    def __init__(self, use_relations, max_leaves, max_conjuncts):
        self.use_relations = use_relations
        self.max_leaves = max_leaves
        self.max_conjuncts = max_conjuncts

    def extract_nodes(self, content):
        nodes = {}
        for edge in content:
            rel, w1, i1, w2, i2 = EDGE_RE.match(edge.replace("'", "")).groups()
            i1 = int(i1)
            i2 = int(i2)
            w2, t2 = w2.rsplit("/", 1)
            node = Node(w2.lower(), t2, i1, rel, [])
            if i2 in nodes:
                if precedence(node.rel) < precedence(nodes[i2].rel):
                    nodes[i2] = node
            else:
                nodes[i2] = node
        return nodes

    def annotate_paths(self, nodes):
        for i, node in nodes.items():
            path = node.path
            at = node
            hit = {i}
            while at.parent in nodes:
                if "nmod:" in at.rel:
                    path.append(at.rel.split(":")[1])
                if at.parent in hit:
                    break
                hit.add(at.parent)
                at = nodes[at.parent]

    def extract_predicates(self, nodes):
        preds = []
        for i, node in sorted(nodes.items()):
            if not CONTENT_RE.match(node.tag):
                continue
            if re.match(BE_FORMS, node.word):
                continue
            pred = node.word
            if len(node.path) > 0 and self.use_relations:
                pred = "(%s %s)" % (node.path[0], pred)
            preds.append(pred)
        return list(set(preds))

    def make_lfs(self, wh, content):
        nodes = self.extract_nodes(content)
        self.annotate_paths(nodes)
        predicates = self.extract_predicates(nodes)
        if self.max_leaves is not None:
            predicates = predicates[:self.max_leaves]
        out = []
        for i in range(1, max(self.max_conjuncts + 1, len(predicates))):
            comb = itertools.combinations(predicates, i)
            for pred_comb in comb:
                if len(pred_comb) == 1:
                    out.append("(%s %s)" % (wh, pred_comb[0]))
                else:
                    out.append("(%s (and %s))" % (wh, " ".join(pred_comb)))
        return out

    def parse_all(self, stream):
        queries = []
        query_lines = []
        got_question = False
        question = None
        for line in stream:
            sline = line.strip()
            if sline == "" and not got_question:
                got_question = True
                question = query_lines[0].lower()
                query_lines = []
            elif sline == "":
                got_question = False

                queries = None
                for expr in WH_RES:
                    m = re.match(expr, question)
                    if m is None:
                        continue
                    wh = m.group(1).replace(" ", "_")
                    if re.match(BE_FORMS, wh):
                        wh = "is"
                    n_expr_words = len(expr.split(" "))
                    content = query_lines[n_expr_words:]
                    queries = self.make_lfs(wh, content)
                    success = True
                    break

                if not queries:
                    queries = ["(_what _thing)"]

                yield queries
                query_lines = []

            else:
                query_lines.append(sline)
                
"""
This script consumes output from the Stanford parser on stdin. I run the parser as

    java -mx150m -cp "$scriptdir/*:" edu.stanford.nlp.parser.lexparser.LexicalizedParser \
     -outputFormat "words,typedDependencies" -outputFormatOptions "stem,collapsedDependencies,includeTags" \
     -sentences newline \
     edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz \
     $*
"""

if __name__ == "__main__":
    #parser = LfParser(use_relations=True, max_conjuncts=2, max_leaves=None)
    parser = LfParser(use_relations=False, max_conjuncts=2, max_leaves=2)
    for parses in parser.parse_all(sys.stdin):
        print ";".join(parses)

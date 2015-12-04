#!/usr/bin/env python2

import sexpdata

def parse_tree(p):
    if "'" in p:
        p = "none"
    parsed = sexpdata.loads(p)
    extracted = extract_parse(parsed)
    return extracted

def extract_parse(p):
    if isinstance(p, sexpdata.Symbol):
        return p.value()
    elif isinstance(p, int):
        return str(p)
    elif isinstance(p, bool):
        return str(p).lower()
    elif isinstance(p, float):
        return str(p).lower()
    return tuple(extract_parse(q) for q in p)

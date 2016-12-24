#!/usr/bin/env python2

import re
import sys

def name_to_match(name, used_names):
  if isinstance(name, int) and name not in used_names:
    r_name = "(?P<g%s>[/\\w-]+)" % name
    used_names.add(name)
  elif isinstance(name, int):
    r_name = "(?P=g%s)" % name
  else:
    r_name = name
  return r_name

def render(spec, match):
  if isinstance(spec, str):
    return spec
  if isinstance(spec, int):
    rel = match.group("rel")
    token = match.group("g%d" % spec)
    word = token.split("/")[0]
    if "subj" in rel:
        #return "subj__" + word
        return word
    elif "obj" in rel:
        #return "obj__" + word
        return word
    else:
        return word

  return "(%s)" % " ".join([render(p, match) for p in spec])

def match_simple_query(query, edges):
  edge_spec, query_spec = query
  used_names = set()
  query_re = ".*"
  for rel, head, tail in edge_spec:
    r_head = name_to_match(head, used_names)
    r_tail = name_to_match(tail, used_names)
    query_re += "(?P<rel>%s)\(%s, %s\).*" % (rel, r_head, r_tail)

  m = re.match(query_re, edges.lower())
  if m is None:
    return None

  return render(query_spec, m)

verb_query_1 = (
  [ 
    ("(nsubj|dobj|det|dep|nsubjpass|acl:relcl)", 0, r"what/w[a-z]+-\d+"),
  ],
  ("what", 0)
)

verb_query_2 = (
  [ 
    ("(nsubj|dobj|det|dep|nsubjpass|acl:relcl)", r"what/w[a-z]+-\d+", 0),
  ],
  ("what", 0)
)

SIMPLE_QUERIES = [
    verb_query_1,
    verb_query_2
]
def make_simple_query(edges):
  for query in SIMPLE_QUERIES:
    m = match_simple_query(query, edges)
    if m: break
  return m

#QUERY_RE = r"(.*)\((.*)/\w+-([\d\']+), (.*)/\w+-([\d\']+)\)"
#FORBIDDEN_RELS = ["acl:relcl"]
def make_what_query(query_lines):
  joined_lines = "".join(query_lines)
  q = make_simple_query(joined_lines)
  return q

CONTENT_RE = r"([^/]+)/(NN|VB|JJ)"

if __name__ == "__main__":
  queries = []
  query_lines = []
  got_question = False
  question = None
  for line in sys.stdin:
    sline = line.strip()
    if sline == "" and not got_question:
      got_question = True
      question = query_lines[0].lower()
      query_lines = []
    elif sline == "":
      got_question = False

      #print
      #print question

      #if "what is the color of the" in question:
      #  obj = re.search(r", (.*)/", query_lines[6]).group(1)
      #  print "(color %s)" % obj
      m = re.match(r"^what (\w+) (is|are)", question)
      if m is not None:
        wh = m.group(1)
        content = query_lines[3:]
        content = [w.split()[1] for w in content]
        content = [re.match(CONTENT_RE, w) for w in content]
        content = [m for m in content if m]
        if len(content) == 0:
          target = "object"
        else:
          target = content[0].group(1).lower()

        print "(%s %s)" % (wh, target)
        query_lines = []
        continue

      m = re.match(r"^what (is|are) the (\w+) of", question)
      if m is not None:
        wh = m.group(1)
        content = query_lines[5:]
        content = [w.split()[1] for w in content]
        content = [re.match(CONTENT_RE, w) for w in content]
        content = [m for m in content if m]
        if len(content) == 0:
          target = "object"
        else:
          target = content[0].group(1).lower()

        print "(%s %s)" % (wh, target)
        query_lines = []
        continue

      m = re.match(r"^what (\w+) of", question)
      if m is not None:
        wh = m.group(1)
        content = query_lines[3:]
        content = [w.split()[1] for w in content]
        content = [re.match(CONTENT_RE, w) for w in content]
        content = [m for m in content if m]
        if len(content) == 0:
          target = "object"
        else:
          target = content[0].group(1).lower()

        print "(%s %s)" % (wh, target)
        query_lines = []
        continue

      m = re.match(r"^what", question)
      if m is not None:
        content = query_lines[1:]
        content = [w.split()[1] for w in content]
        content = [re.match(CONTENT_RE, w) for w in content]
        content = [m for m in content if m]
        content = [m for m in content if m.group(1) not in ("be", "do", "have")]
        if len(content) == 0:
          target = "object"
        else:
          target = content[0].group(1).lower()

        print "(what %s)" % target
        query_lines = []
        continue

      m = re.match(r"^(is|are|has|have|were|did|does|do|was) ", question)
      if m is not None:
        wh = m.group(1)
        content = query_lines[1:]
        content = [w.split()[1] for w in content]
        content = [re.match(CONTENT_RE, w) for w in content]
        content = [m for m in content if m]
        if len(content) == 0:
          print "none"
        elif len(content) == 1:
          print "(is1 %s)" % content[0].group(1).lower()
        else:
          print "(is2 %s %s)" % (content[0].group(1).lower(), content[1].group(1).lower())

        query_lines = []
        continue

      m = re.match(r"^how many", question)
      if m is not None:
        content = query_lines[2:]
        content = [w.split()[1] for w in content]
        content = [re.match(CONTENT_RE, w) for w in content]
        content = [m for m in content if m]
        if len(content) == 0:
          target = "object"
        else:
          target = content[0].group(1).lower()

        print "(how_many %s)" % target
        query_lines = []
        continue

      m = re.match(r"^where", question)
      if m is not None:
        content = query_lines[2:]
        content = [w.split()[1] for w in content]
        content = [re.match(CONTENT_RE, w) for w in content]
        content = [m for m in content if m]
        if len(content) == 0:
          target = "object"
        else:
          target = content[0].group(1).lower()

        print "(where %s)" % target
        query_lines = []
        continue

      m = re.match(r"^(can|could) ", question)
      if m is not None:
        wh = m.group(1)
        content = query_lines[1:]
        content = [w.split()[1] for w in content]
        content = [re.match(CONTENT_RE, w) for w in content]
        content = [m for m in content if m]
        if len(content) == 0:
          print "(what object)"
        elif len(content) == 1:
          print "(can1 %s)" % content[0].group(1).lower()
        else:
          print "(can2 %s %s)" % (content[0].group(1).lower(), content[1].group(1).lower())

        query_lines = []
        continue

      print "(what object)"
      query_lines = []
      continue


    #####

      if "how many" in question:
        mtch = None
        idx = 2
        while mtch is None and idx < len(query_lines):
          mtch = re.search(r", (.*)/N", query_lines[idx])
          idx += 1
        if mtch is None:
          mtch = re.search(r", (.*)/", query_lines[2])
        assert mtch is not None
        obj = mtch.group(1)
        print "(count %s)" % obj

      elif "where" in question:
        mtch = None
        idx = 2
        while mtch is None and idx < len(query_lines):
          mtch = re.search(r", (.*)/N", query_lines[idx])
          idx += 1
        if mtch is None and len(query_lines) >= 3:
          mtch = re.search(r", (.*)/", query_lines[2])
        if mtch is not None:
          obj = mtch.group(1)
        else:
          obj = "object"
        print "(where %s)" % obj

      else:
        query = make_what_query(query_lines)
        if query is None:
          #print "\n".join(query_lines)
          #print "warning: null query"
          query = "(_what _thing)"
        print query

      #query = convert_to_query(query_lines)
      #queries.append(query)

      #print question
      #print "\n".join(query_lines)
      #if query is None:
      #  print "none"
      #else:
      #  print query
      #print

      query_lines = []
    else:
      query_lines.append(sline)

  #print len(queries)

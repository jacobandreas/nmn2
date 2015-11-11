class Struct:
  def __init__(self, **entries):
    rec_entries = {}
    for k, v in entries.items():
      if isinstance(v, dict):
        rv = Struct(**v)
      elif isinstance(v, list):
        rv = []
        for item in v:
          if isinstance(item, dict):
            rv.append(Struct(**item))
          else:
            rv.append(item)
      else:
        rv = v
      rec_entries[k] = rv
    self.__dict__.update(rec_entries)

  def __str_helper(self, depth):
    lines = []
    for k, v in self.__dict__.items():
      if isinstance(v, Struct):
        v_str = v.__str_helper(depth + 1)
        lines.append("%s:\n%s" % (k, v_str))
      else:
        lines.append("%s: %r" % (k, v))
    indented_lines = ["  " * depth + l for l in lines]
    return "\n".join(indented_lines)

  def __str__(self):
    return "struct {\n%s\n}" % self.__str_helper(1)

  def __repr__(self):
    return "Struct(%r)" % self.__dict__

class Index:
  def __init__(self):
    self.contents = dict()
    self.ordered_contents = []
    self.reverse_contents = dict()

  def __getitem__(self, item):
    if item not in self.contents:
      return None
    return self.contents[item]

  def get_or_else(self, item, alt):
    res = self[item]
    return alt if res is None else res

  def index(self, item):
    if item not in self.contents:
      idx = len(self.contents)
      self.ordered_contents.append(item)
      self.contents[item] = idx
      self.reverse_contents[idx] = item
    return self[item]

  def get(self, idx):
    return self.reverse_contents[idx]

  def __len__(self):
    return len(self.contents)

  def __iter__(self):
    return iter(self.ordered_contents)

def flatten(lol):
    if isinstance(lol, tuple) or isinstance(lol, list):
        return sum([flatten(l) for l in lol], [])
    else:
        return [lol]

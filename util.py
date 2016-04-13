class Index:
  def __init__(self):
    self.contents = dict()
    self.ordered_contents = []
    self.reverse_contents = dict()

  def __getitem__(self, item):
    if item not in self.contents:
      return None
    return self.contents[item]

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

def chunks(l, n):
  """Yield successive n-sized chunks from l."""
  for i in xrange(0, len(l), n):
    yield l[i:i+n]

def strictChunks(l, n):
  for i in xrange(0, len(l), n):
    chunk = l[i:i+n]
    # TODO(jda) not just wrap
    if len(chunk) < n:
      chunk += l[:n-len(chunk)]
    yield chunk

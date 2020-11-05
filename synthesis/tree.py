class Node:
    def __init__(self, type, name, data=None, parent=None, children=None):
        self.type = type  # name of TERM/RULE
        self.data = data  # terminal string/None for rule
        self.is_term = self.data is not None
        self.parent = parent
        self.depth = 1
        self.size = 1
        if children:
            self.children = children
        else:
            self.children = []
        for c in self.children:
            self.depth = max(self.depth, c.depth + 1)
            self.size += c.size


def nodes_by_type(tree: Node):
    ret = {tree.type: [tree]}
    if tree.data is None:
        return ret
    for c in tree.children:
        tmp = nodes_by_type(c)
        for k, v in tmp.items():
            if k in ret.keys():
                ret[k].extend(v)
            else:
                ret[k] = v
    return ret

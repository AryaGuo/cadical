from lark import Tree, Token

from synthesis.main import *


class Node:
    def __init__(self, obj, parent, index):
        if type(obj) == Tree:
            self.is_token = False
            self.data = obj.data
            self.children = obj.children
        else:
            assert type(obj) == Token
            self.is_token = True
            self.type = obj.type
            self.value = obj.value
        self.parent = parent
        self.index = index
        if self.parent is None:
            self.depth = 1
        else:
            self.depth = self.parent.depth + 1

    def update_subtree(self):
        if self.parent is None:
            self.depth = 1
        else:
            self.depth = self.parent.depth + 1
        if not self.is_token:
            for c in self.children:
                c.update_subtree()

    def build_type_dict(self, d: dict, condition):
        if self.is_token:
            if condition(self.type):
                d[self.type].append(self)
            return
        d[self.data].append(self)
        for c in self.children:
            c.build_type_dict(d, condition)

    def subtree(self, condition):
        if self.is_token:
            ret = [self] if condition(self.type) else []
            return ret
        ret = [self] if self.data != 'start' else []
        for c in self.children:
            ret += c.subtree(condition)
        return ret

    @staticmethod
    def convert_tree(tree, parent, index):
        if not cfg.STGP or type(tree) == Node:
            return tree
        cur = Node(tree, parent, index)
        if type(tree) == Tree:
            child_list = []
            for i, c in enumerate(tree.children):
                child_list.append(Node.convert_tree(c, cur, i))
            cur.children = child_list
        return cur

    def __repr__(self):
        if self.is_token:
            return 'Token(%r, %r)' % (self.type, self.value)
        else:
            return 'Tree(%r, %r)' % (self.data, self.children)

import collections
from copy import deepcopy

from lark import Lark, Token, Tree

from synthesis.Node import Node
from synthesis.main import *

Rule_entry = collections.namedtuple('Rule_entry', ['name', 'op', 'is_name'])


class Rule:
    def __init__(self, name, is_token):
        self.name = name
        self.is_token = is_token
        self.prod = []  # [name, OP, is_name]

    def get_weight(self):
        sum = 1
        for entry in self.prod:
            if entry.name in cfg.wt:
                sum += cfg.wt[entry.name]
        return sum

    def __repr__(self):
        return '{}[{}]'.format(self.name, self.prod)


class DSL:
    def __init__(self, meta_parser, grammar_file):
        self.grammar_file = grammar_file
        with open(grammar_file) as grammar:
            self.parser = Lark(grammar)
        with open(grammar_file) as grammar:
            self.grammar_tree = meta_parser.parse(grammar.read())
            self.single_terminals, self.rule_dict, self.token_dict = self.__init_dicts(self.grammar_tree)
        self.grow_type_table, self.full_type_table = self.__build_type_table()

    def variable_token(self, token_type):
        return token_type in self.token_dict and token_type not in self.single_terminals

    def gen_random_scheme(self, rule, depth):
        return self.get_scheme_from_dsl(self.__gen_random_dsl(rule, depth), False)

    def get_scheme_from_dsl(self, dsl, test_mode, name=None):
        scheme = Scheme(dsl, self.parser.parse(dsl), test_mode, name)
        return scheme

    def __gen_random_dsl(self, name: str, depth: int, strategy: str = 'grow'):
        """
        :param name: Name of the generated rule.
        :param depth: Current depth limit.
        :param strategy: 'grow' or 'full'
        :return: DSL in string.
        """
        if depth < 0:
            return None
        ret = ''
        type_table = self.full_type_table if strategy == 'full' else self.grow_type_table
        allowed_rules = type_table[depth][name]
        if len(allowed_rules) == 0:
            return None
        rule = random.choices(allowed_rules, [rule.get_weight() for rule in allowed_rules])[0]
        for entry in rule.prod:
            if not entry.is_name:
                ret += entry.name
            else:
                if entry.op and (entry.name not in type_table[depth - 1] or random.random() > cfg.OP_prob):
                    continue
                if entry.name not in self.rule_dict and entry.name not in self.token_dict:
                    ret += entry.name
                else:
                    ret += self.__gen_random_dsl(entry.name, depth - 1, strategy)
        return ret

    def __gen_random_tree(self, rule, depth, is_token=False):
        dsl_str = self.__gen_random_dsl(rule, depth)
        if dsl_str is None:
            return None

        with open(self.grammar_file) as grammar:
            parser = Lark(grammar)
            if is_token:
                return parser.parse(dsl_str).children[0]
            else:
                return parser.parse(dsl_str, rule)

    def mutate(self, parse_tree, depth):
        if type(parse_tree) == Token:
            if self.variable_token(parse_tree.type):
                return self.__gen_random_tree(parse_tree.type, cfg.depth_lim - 1, True)
            else:
                return parse_tree

        assert type(parse_tree) == Tree, parse_tree

        p_stop = depth / (depth + 8)
        if random.random() < p_stop and parse_tree.data != 'start':
            return self.__gen_random_tree(parse_tree.data, cfg.depth_lim - 1)

        rand_child_index = random.randrange(len(parse_tree.children))
        parse_tree.children[rand_child_index] = self.mutate(parse_tree.children[rand_child_index], depth + 1)
        return parse_tree

    def mutate_(self, tree):
        nodes = tree.subtree(self.variable_token)
        for _ in range(cfg.gen_restart):
            rand_node = random.choice(nodes)
            if rand_node.is_token:
                other = self.__gen_random_tree(rand_node.type, cfg.depth_lim - rand_node.depth, True)
            else:
                other = self.__gen_random_tree(rand_node.data, cfg.depth_lim - rand_node.depth)
            if other is not None:
                other = Node.convert_tree(other, rand_node.parent, rand_node.index)
                break
        else:
            print('mutation failed')
            return tree
        if rand_node.parent:
            rand_node.parent.children[rand_node.index] = other
            return tree
        else:
            return other

    def crossover(self, tree_a, tree_b, depth):
        """
        Randomly substitute a branch of tree_a with a branch of tree_b.
        :param tree_a:
        :param tree_b:
        :param depth: Current depth in the tree. Used to calculate probability.
        :return: Tree_a' with new branch.
        """

        def __match_node(tree_a, tree_b):
            if type(tree_a) == Token:
                return type(tree_b) == Token and tree_a.type == tree_b.type and self.variable_token(tree_a.type)
            else:
                return type(tree_b) == Tree and tree_a.data == tree_b.data

        # print('crossover @ depth', depth, tree_a.data if type(tree_a) == Tree else tree_a)
        if type(tree_a) == Token:
            assert type(tree_b) == Token, tree_b
            return tree_b

        assert type(tree_a) == Tree, tree_a
        assert type(tree_b) == Tree, tree_b
        assert tree_a.data == tree_b.data, (tree_a, tree_b)

        p_stop = depth / (depth + 8)
        if random.random() < p_stop and tree_a.data != 'start':
            return tree_b

        matched = []
        for i in range(len(tree_a.children)):
            for j in range(len(tree_b.children)):
                if __match_node(tree_a.children[i], tree_b.children[j]):
                    matched.append((i, j))
        if len(matched) > 0:
            n = random.randrange(len(matched))
            tree_a.children[matched[n][0]] = self.crossover(tree_a.children[matched[n][0]],
                                                            tree_b.children[matched[n][1]], depth + 1)
            return tree_a
        else:
            return tree_b

    def crossover_(self, tree_a, tree_b):
        dict_a = collections.defaultdict(list)
        dict_b = collections.defaultdict(list)
        tree_a.build_type_dict(dict_a, self.variable_token)
        tree_b.build_type_dict(dict_b, self.variable_token)
        mutual_types = list(frozenset(dict_a) & frozenset(dict_b))
        if not mutual_types:
            raise Exception('No mutual types')
        chosen_type = random.choice(mutual_types)
        swap_a = random.choice(dict_a[chosen_type])
        swap_b = deepcopy(random.choice(dict_b[chosen_type]))
        swap_b.parent, swap_b.index = swap_a.parent, swap_a.index
        swap_b.update_subtree()
        if swap_a.parent:
            swap_a.parent.children[swap_a.index] = swap_b
            return tree_a
        else:
            return swap_b

    @staticmethod
    def parse_rule(tree: Tree):
        ret = []
        assert tree.data == 'rule' or tree.data == 'token', tree.data
        expansions = tree.children[-1]
        if expansions.data == 'expansions':
            expansion_list = expansions.children
        else:
            expansion_list = [expansions]
        for expansion in expansion_list:
            rule = Rule(tree.children[0].lstrip('?!'), tree.data == 'token')
            if expansion.data == 'expansion':
                exprs = expansion.children
            else:
                exprs = [expansion]
            for expr in exprs:
                if expr.data == 'expr':
                    value, op = expr.children[0], expr.children[-1]
                else:
                    value, op = expr, None
                if value.data == 'name':
                    rule.prod.append(Rule_entry(value.children[0], op, True))
                else:
                    rule.prod.append(Rule_entry(value.children[0].strip('"'), op, False))
            ret.append(rule)
        return ret

    @staticmethod
    def __init_dicts(tree):
        single_terminals = set()
        rule_dict = collections.defaultdict(list)
        token_dict = collections.defaultdict(list)

        for rule in tree.children:
            if rule.data == 'rule':
                key = rule.children[0].lstrip('?!')
                rule_dict[key] = DSL.parse_rule(rule)
            elif rule.data == 'token':
                key = rule.children[0]
                token_dict[key] = DSL.parse_rule(rule)
                if rule.children[-1].data == 'literal':
                    single_terminals.add(key)
        return single_terminals, rule_dict, token_dict

    def __build_type_table(self):
        # key: name; value: Rule
        grow_type_table = [collections.defaultdict(list) for _ in range(cfg.depth_lim)]
        full_type_table = [collections.defaultdict(list) for _ in range(cfg.depth_lim)]
        for token in self.token_dict.keys():
            for rule in self.token_dict[token]:
                grow_type_table[0][token].append(rule)
                full_type_table[0][token].append(rule)
        for i in range(1, cfg.depth_lim):
            for k in grow_type_table[i - 1].keys():
                grow_type_table[i][k].extend(grow_type_table[i - 1][k])
            for name in self.rule_dict.keys():
                for rule in self.rule_dict[name]:
                    grow_check = [
                        not rule.prod[j].is_name or rule.prod[j].op or rule.prod[j].name in grow_type_table[i - 1]
                        for j in range(len(rule.prod))]
                    full_check = [
                        not rule.prod[j].is_name or rule.prod[j].op or rule.prod[j].name in full_type_table[i - 1]
                        for j in range(len(rule.prod))]
                    if all(grow_check):
                        grow_type_table[i][name].append(rule)
                    if all(full_check):
                        full_type_table[i][name].append(rule)
        return grow_type_table, full_type_table

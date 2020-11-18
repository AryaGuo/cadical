class Config:
    def __init__(self):
        # file paths
        self.output_root = '../result'
        self.meta_file = 'bnf.bnf'
        self.grammar_file = 'expr.bnf'

        # GP params
        self.pop_size = 20
        self.depth_lim = 10
        self.tournament_size = 2
        self.elitism = True
        self.seed = None
        self.epoch = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.9
        self.gen_restart = 50

        # checkpoint
        self.save = 1
        self.report = 5

        # evaluation
        self.time_lim = 60
        self.threshold = 60
        self.eval = None

        # method
        self.ratio = False
        self.STGP = True
        self.monkeys = None  # 'grammars.typed'
        self.load = None

        self.wt = dict()
        self.wt['LHS'] = 1

        self.OP_prob = 0.7


cfg = Config()

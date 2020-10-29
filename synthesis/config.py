class Config:
    meta_file = 'bnf.bnf'
    grammar_file = 'expr.bnf'
    pop_size = 10
    depth_lim = 10
    tournament_size = 4
    elitism = True
    seed = None
    epoch = 2
    mutation_rate = 0.1

    wt = dict()
    wt['NON_ZERO'] = 5
    wt['NUMBER'] = 5
    wt['NEW_SCORE'] = 1
    wt['SCORE_INC'] = 2
    wt['CONFLICT_INDEX'] = 3
    wt['SCORE'] = 2
    wt['+'] = wt['-'] = wt['*'] = wt['/'] = wt['^'] = 1
    OP_prob = 0.7

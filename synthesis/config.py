class Config:
    meta_file = 'bnf.bnf'
    grammar_file = 'expr.bnf'
    pop_size = 10
    depth_lim = 10
    tournament_size = 4
    elitism = True
    seed = None
    epoch = 30
    mutation_rate = 0.3
    gen_restart = 50
    save = 1
    report = 5

    wt = dict()
    # wt['NUMBER'] = 5
    # wt['DECIMAL'] = 5
    # wt['POWER'] = 2
    # wt['CONFLICT_INDEX'] = 3
    # wt['+'] = wt['-'] = wt['*'] = wt['/'] = wt['^'] = 1
    OP_prob = 0.7

class Config:
    meta_file = 'bnf.bnf'
    grammar_file = 'expr.bnf'
    pop_size = 20
    depth_lim = 10
    tournament_size = 2
    elitism = True
    seed = None
    epoch = 20
    mutation_rate = 0.2
    crossover_rate = 0.9
    gen_restart = 50
    save = 1
    report = 5
    time_lim = 60

    wt = dict()
    wt['LHS'] = 1

    OP_prob = 0.7

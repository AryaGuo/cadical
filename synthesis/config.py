class Config:
    meta_file = 'bnf.bnf'
    grammar_file = 'expr.bnf'
    pop_size = 10
    depth_lim = 10
    tournament_size = 4
    elitism = False
    seed = 22
    epoch = 5
    mutation_rate = 0.1
# AssertionError: ([Tree('assign_unbumped', [Token('NEW_SCORE', 'new_score'), Token('EQUAL', '='), Tree('ub_div', [Tree('ub_mul', [Tree('ub_score', [Token('SCORE', 's')]), Token('STAR', '*'), Tree('ub_idx', [Token('CONFLICT_INDEX', 'i')])]), Token('SLASH', '/'), Tree('ub_idx', [Token('CONFLICT_INDEX', 'i')])]), Token('SEMICOLON', ';')]),
# Tree('assign_new_score', [Token('NEW_SCORE', 'new_score'), Token('EQUAL', '='), Tree('ns_score', [Token('SCORE', 's')]), Token('SEMICOLON', ';')]),
# Tree('assign_score_inc', [Tree('condition', [Token('IF', 'if'), Token('LPAR', '('), Tree('bool', [Token('NUMBER', '3'), Tree('equal', [Token('__ANON_0', '==')]), Token('NUMBER', '7')]), Token('RPAR', ')')]), Token('SCORE_INC', 'score_inc'), Token('EQUAL', '='), Tree('inc_sub', [Tree('inc_add', [Tree('inc_idx', [Token('CONFLICT_INDEX', 'i')]), Token('PLUS', '+'), Tree('inc_inc', [Token('SCORE_INC', 'score_inc')])]), Token('MINUS', '-'), Tree('inc_mul', [Tree('inc_idx', [Token('CONFLICT_INDEX', 'i')]), Token('STAR', '*'), Tree('inc_inc', [Token('SCORE_INC', 'score_inc')])])]), Token('SEMICOLON', ';')])],
# [Tree('assign_new_score', [Token('NEW_SCORE', 'new_score'), Token('EQUAL', '='), Tree('ns_add', [Tree('ns_add', [Token('NUMBER', '0'), Token('PLUS', '+'), Tree('ns_div', [Tree('ns_inc', [Token('SCORE_INC', 'score_inc')]), Token('SLASH', '/'), Token('NUMBER', '7')])]), Token('PLUS', '+'), Tree('ns_mul', [Tree('ns_inc', [Token('SCORE_INC', 'score_inc')]), Token('STAR', '*'), Token('NUMBER', '3')])]), Token('SEMICOLON', ';')])])

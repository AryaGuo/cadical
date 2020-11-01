static = """
        new_score = s;
    """
inc = """
        new_score = s + 1;
    """
sum = """
        new_score = s + i;
    """
vsids = """
        if (i % (2^8) == 0)
            new_score = s / 2;
        new_score = s + 1;
    """
nvsids = """
        old_score = 0.8 * s;
        new_score = s + (1 - 0.8);
    """
evsids = """
        new_score = s + (1/0.8)^i;
    """
acids = """
        new_score = (s + i) / 2;
    """
vmtf = """
        new_score = i;
    """

w37 = """
        new_score = s + 2 - 6 / i * i + score_inc; 
"""

w37eq = """
        new_score = s - 4 + score_inc;
"""

w32 = """
        new_score = 4 + i / 2 + score_inc / score_inc * s;
        if (i != i)
            score_inc = score_inc / 4 / score_inc;
"""

w32eq = """
        new_score = 4 + i / 2 + s; 
"""
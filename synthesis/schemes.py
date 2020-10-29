static = """"""
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
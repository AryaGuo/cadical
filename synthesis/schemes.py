static = """
        bumped = lhs;
    """
inc = """
        bumped = lhs + 1;
    """
sum = """
        bumped = lhs + i;
    """
vsids = """
        if (i % 256 == 0)
            unbumped = lhs * 0.5;
        bumped = lhs + 1;
    """
nvsids = """
        unbumped = 0.8 * lhs;
        bumped = lhs + (1 - 0.8);
    """
evsids = """
        bumped = lhs + inc;
        inc = lhs * (1 / 0.8);
    """
acids = """
        bumped = (lhs + i) / 2;
    """
vmtf = """
        bumped = i;
    """

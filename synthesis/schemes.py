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
        ninc = lhs * (1 / 0.8);
    """
acids = """
        bumped = (lhs + i) / 2;
    """
vmtf = """
        bumped = i;
    """
gen1 = """
        if (i == 512)
            unbumped = inc * 0.9 + 1 / 128;
        bumped = 2 * 0.4 + i;
        if (i >= i)
            ninc = lhs;
"""

hard_37 = """
        if (128 + i % i == i * i)
            unbumped = lhs / i + i + i * i * lhs;
        bumped = lhs + inc;
        if (32 >= 6)
            ninc = lhs + lhs / 256; 
"""

dsl_36 = """
        bumped = inc / i + i / i / 0.9;
        ninc = i / 8 + inc;
"""

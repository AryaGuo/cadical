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

stgp_37 = """
        if (512 == i)
            unbumped = i - lhs * i;
        bumped = i;
        if (4 == 512)
            ninc = i;
"""

stgp_hard_37 = """
        if (i >= i)
            unbumped = lhs * 1024;
        bumped = lhs + inc;
        ninc = 64 + 16;
"""

monkeys = """
        unbumped = (lhs + 2) * ((0.4 / 2) / 9);
        bumped = 0.9;
        ninc = inc;
"""

ratio = """
        bumped = 32 - i + inc / 64 * inc;
        if ( i >= 5 )
            ninc = lhs + 0 - 0.3 ;
"""

par_37 = """
        bumped = i + i / 16 + lhs - 0.5 * lhs / 6;
        if ( i * 1024 == 0)
            ninc = inc / 256 / 0.9 - i;
"""

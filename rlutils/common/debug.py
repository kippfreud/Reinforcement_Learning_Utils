"""
Functions for debugging.
"""

def print_graph(g, level=0):
    """Print computation graph for a grad_fn."""
    if g == None: return
    print(" "*level*4, g)
    for subg in g.next_functions:
        print_graph(subg[0], level+1)
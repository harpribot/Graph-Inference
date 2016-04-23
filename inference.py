from graph.graph import Graph
import numpy as np

G = Graph()

# add variable nodes
a = G.add_var_node('a',2)
b = G.add_var_node('b',3)

# connecting factor
P_ba = np.array([[0.2, 0.8], [0.8, 0.2], [0.6, 0.4]])
G.add_factor_node(P_ba, b, a)
# run sum-product and get marginals for variables
G.compute_all_marginals()
marginal_dict = G.get_all_marginals()
print 'distribution (a) :'
print marginal_dict['a']
print 'distribution (b) :'
print marginal_dict['b']

# reset before altering graph further
G.reset_all_messages()
################ UPON EXPLICIT CONDITIONING #############

# condition on variables
G.vars['a'].condition(1)

# Now compute all marginals using sum product
G.compute_all_marginals()
marginal_dict = G.get_all_marginals()
print 'distribution (a) :'
print marginal_dict['a']
print 'distribution (b) :'
print marginal_dict['b']

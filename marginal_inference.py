from graph.graph import Graph
import numpy as np

graph = Graph()

# add variable nodes
G = graph.add_var_node('Grade',2)
D = graph.add_var_node('Course Difficulty',2)
I = graph.add_var_node('Intelligence', 2)
# connecting factor
P_G_DI = np.array([[[0.5,0.1],[0.9,0.4]],[[0.5,0.9],[0.1,0.6]]])
graph.add_factor_node(P_G_DI, G, D, I)
# run sum-product and get marginals for variables
graph.compute_all_marginals()
marginal_dict = graph.get_all_marginals()
print 'distribution (Grade) :'
print marginal_dict['Grade']

# reset before altering graph further
graph.reset_all_messages()
################ UPON EXPLICIT CONDITIONING #############

# condition on variables
graph.vars['Intelligence'].condition(1)
graph.vars['Course Difficulty'].condition(0)
# Now compute all marginals using sum product
graph.compute_all_marginals()
marginal_dict = graph.get_all_marginals()
print 'distribution (Grade) :'
print marginal_dict['Grade']

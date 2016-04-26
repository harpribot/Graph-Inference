from graph.graph import Graph
import numpy as np
from distribution.data_distribution import get_distribution, reconstruct_dist,get_log_likelihood
from scipy.stats import rv_discrete
from distribution.create_data import Creator

'''
    Write now due to time constraints the get_distribution, reconstruct_dist
    and get_log_likelihood functions are hard coded for the graph in this example
    . I CLASSified (general implementation :P) the create_data class, and all
    of the belief propogation method, but the above 3 methods are not general yet.
    If i get time, I will improve the codes to handle any general graph, but the
    data_distribution helper functions are hard coded. However they are general
    enough that you can alter them to handle any graph
'''
################ CREATE DATA ###################
# Sampling distribution
P_DI_G = np.array([
                   [[0.25,0.5,0.25],[0.2,0.4,0.4],[0.2,0.2,0.6]],
                   [[0.3,0.6,0.1],[0.25,0.5,0.25],[0.2,0.3,0.5]],
                   [[0.8,0.1,0.1],[0.6,0.2,0.2],[0.5,0.25,0.25]],
                 ])
P_I_S = np.array([[0.7,0.2,0.1],[0.4,0.4,0.2],[0.1,0.1,0.8]])
P_G_R = np.array([[0.7,0.2,0.1],[0.5,0.25,0.25],[0.2,0.3,0.5]])

P_D = np.array([[0.2],[0.4],[0.4]])
P_I = np.array([[0.2],[0.6],[0.2]])

# Create the graph
num_samples = 1001
creator = Creator(num_samples)
creator.add_variable('D', np.arange(3))
creator.add_variable('I', np.arange(3))
creator.add_variable('G', np.arange(3))
creator.add_variable('S', np.arange(3))
creator.add_variable('R', np.arange(3))

# This sampling should be in DAG order else it will fail
creator.sample(P_D, 'D')
creator.sample(P_I, 'I')
creator.sample(P_DI_G, 'G', 'D','I') # P(G|D,I)
creator.sample(P_G_R, 'R', 'G') # P(R|G)
creator.sample(P_I_S, 'S', 'I') # P(S|I)

# Store and get the data
outfile = './data/virtual_data.csv'
creator.store_samples(outfile)
data_frame = creator.get_frame()

# True distribution
P_I_true, P_D_true,P_DI_G_true, P_G_R_true, P_I_S_true = get_distribution(data_frame)
# True value of hidden variables
true_G = data_frame['G'].values
# Modify the distribution to move away from desired and lets see if it can recover
P_DI_G = np.array([
                   [[0.3,0.4,0.3],[0.25,0.5,0.25],[0.1,0.1,0.8]],
                   [[0.4,0.4,0.2],[0.3,0.4,0.3],[0.25,0.35,0.4]],
                   [[0.6,0.2,0.2],[0.7,0.15,0.15],[0.4,0.3,0.3]],
                 ])
P_G_R = np.array([[0.5,0.3,0.2],[0.4,0.3,0.3],[0.2,0.2,0.6]])
'''
    Note: We modified P_DI_G and P_G_R only as these are the parameters that are
    to be tuned. These are the only parameters that include Grade (G) and thus
    these are the only unknown CPD tables as Grade is hidden.
'''

######################## Initialize the Factor Graph ######################
graph = Graph()

# add variable nodes
G = graph.add_var_node('Grade',3) # Grade on GPA scale - 0 - 1.5(0)/ 1.5 - 3(1)/3 - 4(2)
D = graph.add_var_node('Course Difficulty',3) # Easy(0)/ Moderate(1) / Hard(2)
I = graph.add_var_node('Intelligence', 3) # Poor(0) / Average(1) / Exceptional(2)
R = graph.add_var_node('Recommondation Letter', 3) # Bad(0) / Average(1) / Excellent(2)
S = graph.add_var_node('SAT Performance', 3) # Poor(0)/ Average(1) / Excellent(1)

for iteration in range(0,10):
    print 'Iteration :', iteration
    # Expectation Step - Initialize Grade using the sum-product algorithm using Bel-Prop
    # Form the factor graph from the present distribution
    graph.add_factor_node(P_DI_G, D, I, G)
    graph.add_factor_node(P_G_R, G, R)
    graph.add_factor_node(P_I_S, I, S)
    graph.add_factor_node(P_D, D)
    graph.add_factor_node(P_I, I)
    # Now the hidden value is Grade
    map_DI_G = {}
    map_DI = {}
    map_R_G = {}
    map_G = {}
    index = 0
    G_val = []
    log_likelihood = 0
    for d,i,s,r in zip(data_frame['D'].values, data_frame['I'].values, \
                    data_frame['S'].values,data_frame['R'].values):
        if(index % 100 == 0):
            print index
        index += 1
        # reset before altering graph further
        graph.reset_all_messages()
        # condition on variables
        graph.vars['Intelligence'].condition(i)
        graph.vars['Course Difficulty'].condition(d)
        graph.vars['Recommondation Letter'].condition(r)
        graph.vars['SAT Performance'].condition(s)
        # compute the marginals to get that for hidden variable - Grade
        graph.compute_all_marginals()
        marginal_dict = graph.get_all_marginals()
        g_dist = marginal_dict['Grade']
        # sample the new values of Grade from this distribution g_dist
        g_range = np.arange(P_DI_G.shape[-1])
        sampler_G_DI = rv_discrete(name='sampler_G_DI',values=(g_range,g_dist))
        # Obtain the new value of G
        g_new = sampler_G_DI.rvs(size=1)[0]
        G_val.append(g_new)

        # Calculate the log likelihood of the data sample and add it to log_likelihood
        log_likelihood += get_log_likelihood(d,i,g_new,s,r, \
                                                P_D, P_I, P_DI_G, P_I_S, P_G_R)
        # Accumulate E#(G ^ D ^ I)
        if (d,i) in map_DI_G:
            map_DI_G[(d,i)] = map_DI_G[(d,i)] + g_dist
        else:
            map_DI_G[(d,i)] = g_dist
        # Accumulate E#(D ^ I)
        if (d,i) in map_DI:
            map_DI[(d,i)] = map_DI[(d,i)] + 1
        else:
            map_DI[(d,i)] = 1
        # Accumulate E#(R ^ G)
        if r in map_R_G:
            map_R_G[r] = map_R_G[r] + g_dist
        else:
            map_R_G[r] = g_dist
        # Accumulate E#(G)
        if 1 in map_G:
            map_G[1] = map_G[1] + g_dist
        else:
            map_G[1] = g_dist

    # Print the log likelihood, Ideally this should increase all the time, and finally saturate
    avg_log_likelihood = log_likelihood/float(num_samples)
    print 'Log likelihood:', avg_log_likelihood


    # Maximization Step - Now take this data, compute the distribution from this data
    P_DI_G, P_G_R  = reconstruct_dist(map_DI_G, map_DI, map_R_G, map_G)

    G_val = np.array(G_val)
    # Get the number of correct matches
    percentage_match = sum(G_val == true_G)/float(num_samples)
    print 'Present Recovery:',percentage_match

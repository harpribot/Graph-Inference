import numpy as np
from node import VarNode, FactorNode

class Graph:
    def __init__(self):
        self.vars = {}
        self.factors = []
        self.dims = []

    def add_var_node(self, name, dim):
        var_id = len(self.vars)
        varNode = VarNode(name, dim, var_id)
        self.vars[name] = varNode
        self.dims.append(dim)

        return varNode


    def add_factor_node(self, joint_prob, *args):
        factor_id = len(self.factors)
        factorNode = FactorNode(joint_prob,factor_id,*args)
        self.factors.append(factorNode)

        return factorNode

    def reset_all_messages(self):
        for var_name, var_node in self.vars.iteritems():
            var_node.reset_node()

        for factor_node in self.factors:
            factor_node.reset_node()

    def graph_inference(self,opn='sum-product',maxsteps=500):
        '''
            This is the algorithm for max-sum inference/ sum-product inference
        '''
        # loop to convergence
        timestep = 0
        while timestep < maxsteps:
            timestep = timestep + 1

            for f in self.factors:
                # start with factor-to-variable
                f.prepare_outgoing_msgs(opn)
                f.send_messages()

            for k, v in self.vars.iteritems():
                # variable-to-factor
                v.prepare_outgoing_msgs(opn)
                v.send_messages()


    def compute_all_marginals(self, maxsteps=500):
        """ Return dictionary of all marginal distributions
            indexed by corresponding variable name
        """
        # Message pass
        self.graph_inference('sum-product', maxsteps)

        self.marginals = {}
        # for each var
        for k, v in self.vars.iteritems():
            # multiply together messages
            vmarg = 1
            for i in range(0, len(v.incoming_msgs)):
                vmarg = vmarg * v.incoming_msgs[i]

            # normalize the marginal
            n = np.sum(vmarg)
            vmarg = vmarg / n

            self.marginals[k] = vmarg


    def get_all_marginals(self):
        return self.marginals

    def compute_best_assignment(self):
        pass

    def get_best_assignment(self):
        pass

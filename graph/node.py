import numpy as np
#from builtins import range
from functools import reduce

class Node(object):
    '''
        Superclass for variable nodes and factor nodes
    '''
    def __init__(self,nid):
        self.node_id = nid
        self.neighbors = []
        self.incoming_msgs = []
        self.outgoing_msgs = []
        self.older_outgoing_msgs = []

    def nextStep(self):
        self.older_outgoing_msgs = self.outgoing_msgs[:]

    def normalize_outgoing_msgs(self):
        '''
            locally normalizes each message to 1 before sending
            this happens bcoz each message is some probability distribution and
            probabilities add to 1
        '''
        self.outgoing_msgs = [msg/np.sum(msg) for msg in self.outgoing_msgs]

    def receive_message(self, sender_node, message):
        '''
            Receives message from the from_node
        '''
        sender_loc_neighbor_lst = self.neighbors.index(sender_node)
        self.incoming_msgs[sender_loc_neighbor_lst] = message

    def send_messages(self):
        for i in range(0, len(self.outgoing_msgs)):
            self.neighbors[i].receive_message(self,self.outgoing_msgs[i])


class VarNode(Node):
    def __init__(self,name, dim,node_id):
        super(VarNode,self).__init__(node_id)
        self.name = name
        self.dim = dim
        # if observed then observed, stores the value of observation, if not
        # observed then it stores -1
        self.observed = -1

    def reset_node(self):
        size = range(0, len(self.incoming_msgs))
        self.incoming_msgs = [np.ones((self.dim,1)) for i in size]
        self.outgoing_msgs = [np.ones((self.dim,1)) for i in size]
        self.older_outgoing_msgs = [np.ones((self.dim, 1)) for i in size]
        self.observed = -1

    def condition(self,observed_val):
        self.observed = observed_val

        for i in range(0, len(self.outgoing_msgs)):
            self.outgoing_msgs[i] = np.zeros((self.dim,1))
            self.outgoing_msgs[i][self.observed] = 1.

        self.nextStep()

    def prepare_outgoing_msgs(self):
        if self.observed < 0 and len(self.neighbors) > 1:
            self.nextStep()

            for i in range(0, len(self.incoming_msgs)):
                all_incoming = self.incoming_msgs[:]
                del all_incoming[i] # deletes incoming message from current neighbor
                self.outgoing_msgs[i] = reduce(np.multiply, all_incoming)

            self.normalize_outgoing_msgs()


class FactorNode(Node):
    def __init__(self, conditional_prob, node_id, *args):
        super(FactorNode,self).__init__(node_id)
        self.conditional_prob = conditional_prob
        self.neighbors = list(args) # variable nodes in same order as dimensions
                                    # in conditional distribution

        # no. of edges
        num_nbrs = len(self.neighbors)
        # of dependencies
        num_dependencies = self.conditional_prob.squeeze().ndim

        # initilize all the messages
        for i in range(0, num_nbrs):
            vertex = self.neighbors[i]
            vertex_dim = vertex.dim # number of values it can take

            # initialize factor
            self.incoming_msgs.append(np.ones((vertex_dim, 1)))
            self.outgoing_msgs.append(np.ones((vertex_dim, 1)))
            self.older_outgoing_msgs.append(np.ones((vertex_dim,1)))

            # append factor to all neighbors
            vertex.neighbors.append(self)
            vertex.incoming_msgs.append(np.ones((vertex_dim, 1)))
            vertex.outgoing_msgs.append(np.ones((vertex_dim, 1)))
            vertex.older_outgoing_msgs.append(np.ones((vertex_dim, 1)))

        # check if the factor dimension matches the neighbors
        assert(num_nbrs == num_dependencies), 'conditional Distribution - Neighbor dimension mismatch'


    def reset_node(self):
        # reset incoming, outgoing, and older_outgoing_msgs from all factor neighbors
        for i in range(0, len(self.neighbors)):
            self.incoming_msgs[i] = np.ones((self.neighbors[i].dim,1))
            self.outgoing_msgs[i] = np.ones((self.neighbors[i].dim,1))
            self.older_outgoing_msgs[i] = np.ones((self.neighbors[i].dim,1))

    def prepare_outgoing_msgs(self):
        '''
            prepare outgoing message by doing sum product on the incoming msg
        '''
        # switch reference for older message
        self.nextStep()

        num_incoming = len(self.incoming_msgs)

        # do tiling in advance
        # roll axes to match shape of newMessage after
        for i in range(0,num_incoming):
            # find tiling size
            nextShape = list(self.conditional_prob.shape)
            del nextShape[i]
            nextShape.insert(0, 1)
            # need to expand incoming message to correct num of dims to tile properly
            prepShape = [1 for x in nextShape]
            prepShape[0] = self.incoming_msgs[i].shape[0]
            self.incoming_msgs[i].shape = prepShape
            # tile and roll
            self.incoming_msgs[i] = np.tile(self.incoming_msgs[i], nextShape)
            self.incoming_msgs[i] = np.rollaxis(self.incoming_msgs[i], 0, i+1)

        for i in range(0, num_incoming):
            all_incoming = self.incoming_msgs[:]
            del all_incoming[i] # remove message from the one to whom to sending
            product_incoming = reduce(np.multiply,all_incoming, self.conditional_prob)
            product_incoming = np.rollaxis(product_incoming, i, 0)
            sum_product_incoming = np.sum(product_incoming, tuple(range(1, num_incoming)))
            self.outgoing_msgs[i] = sum_product_incoming

        # normalize outgoing messages
        self.normalize_outgoing_msgs()

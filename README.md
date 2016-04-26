# Expectation Maximization on Graph

There are two ways you can us this work.
1. Do marginal inference on nodes, when none of the nodes is observed
2. Do marginal inference on nodes, when some of the nodes are observed
3. Do Expectation Maximization on the Conditional Probability Distribution when one node is hidden.


The graph implemented for Case 1, and Case 2:

![Graph for marginal inference](https://github.com/harpribot/Belief-Prop/blob/master/images/graph_marginal.jpg)


The graph implemented for Case 3:

![Graph for EM](https://github.com/harpribot/Belief-Prop/blob/master/images/graph_EM.jpg)

The grey node is the hidden node, the yellow nodes are observable.

## Data (Virtual)
The virtual data is located in /data/virtual_data.csv
## Run Instructions
### For case 1 and case 2:
python marginal_inference.py

### For case 3:
python expectation_maximizer.py


## Note:
1. This work does not handle, at present, the case when many nodes are hidden, as then sum-product algorithm won't work in case of dependence of hidden variables. The belief propogation algorithm is run by sum-product engine, so it handles the case when one node is hidden.

2. At present, the data_distribution.py library, is hardcoded for the graph used in example for expectation_maximizer.py. However you can change it to use it in new graphs. I will definitely get back to this, and correct it once my finals are over.

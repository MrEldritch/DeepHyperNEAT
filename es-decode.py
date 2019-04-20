'''
Based on the ES-Hypernet routines in PurePLES

Steps:
1. Get coordinate maps for input and output layers. 
2. Starting from input layer:
  2.1. Find all hidden sheets that the input layer directly feeds into (check mapping tuples)
  2.2. For each such hidden sheet:
    2.2.1. For each node in the input layer:
      2.2.1.1. Quadtree decomposition of the connection weights from that input node, from the CPPN output corresponding to this particular mapping
      2.2.1.2. Prune quadtree for banding
      2.2.1.3. Assign node IDs to the remaining hidden nodes and add the IDs and their coordinates to the node list for the corresponding hidden layer
      2.2.1.4. Record connections between selected input node and these hidden nodes
      (Repeat until connections have been discovered for all input nodes)
    (Repeat until connections have been discovered from input layer to all directly subsequent layers)
  2.3 Now do the same for all the sheets one step deeper, et cetera. Do not find connections from a sheet until you have identified all connections into that sheet.
3. Work backwards from output layer to identify all nodes in directly previous layers that connect *to* output nodes
4. Prune final node graph; eliminate all nodes that do not have a path to both input & output

Max Greason
'''

import numpy as np
import itertools as it
from activations import ActivationFunctionSet
from phenomes import FeedForwardSubstrate
import time

def decode(cppn, input_coordinates, output_):
    
    # Note for ES-Hyperneat implementation: replace input_dimensions, output_dimensions, sheet_dimensions
    # Replace input & output dimensions with lists of (x,y) tuples that directly give positions of all input and output neurons
    # sheet_dimensions can be removed entirely, as it is not relevant when sheet neuron placement is constructed implicitly
    # replace with parameters for ES-HyperNeat stuff, like initial resolution, max resolution, banding threshold, etc.
    
    '''
    Decodes a CPPN into a substrate.
    cppn             -- CPPN
    input_coordinates -- coordinates of substrate input layer nodes
    output_coordinates -- coordinates of substrate output layer nodes
    '''
   
    # Create input layer coordinate map from specified input dimensions
    x = np.linspace(-1.0, 1.0, input_dimensions[1]) if (input_dimensions[1] > 1) else [0.0]
    y = np.linspace(-1.0, 1.0, input_dimensions[0]) if (input_dimensions[0] > 1) else [0.0]
    input_layer = list(it.product(x,y))
   
    # Create output layer coordinate map from specified output dimensions
    x = np.linspace(-1.0,1.0,output_dimensions) if (output_dimensions > 1) else [0.0]
    y = [0.0]
    output_layer = list(it.product(x,y))
   
    sheet = []
    
    # Create list of mappings to be created between substrate sheets
    connection_mappings = [cppn.nodes[x].cppn_tuple for x in cppn.output_nodes if cppn.nodes[x].cppn_tuple[0] != (1,1)]
    
    # Create substrate representation (dictionary of sheets and their respective coordinate maps)
    hidden_sheets = {cppn.nodes[node].cppn_tuple[0] for node in cppn.output_nodes}
    substrate = {s:sheet for s in hidden_sheets}
    substrate[(1,0)] = input_coordinates
    substrate[(0,0)] = output_coordinates
    substrate[(1,1)] = [(0.0,0.0)]
   
    # Create dictionary of output node IDs to their respective mapping tuples
    cppn_idx_dict = {cppn.nodes[idx].cppn_tuple:idx for idx in cppn.output_nodes}
  
    # Create the substrate
return create_es_substrate(cppn, substrate, connection_mappings, cppn_idx_dict)

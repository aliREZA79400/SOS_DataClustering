import numpy as np
from mealpy.utils.space import FloatVar
import os 
import sys
# Get the parent directory
parent_dir = os.path.dirname(os.path.realpath("utils"))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from utils import validators

def generate(K:int,dataset:None):
    '''
    Args:
    K = number of clusters
    '''

    validators.validate_integer(arg_name="number of clusters",
                                arg_value=K,
                                min_value=2,
                                max_value=10)

    data_sorted = np.sort(dataset, axis=0)

    lbs = data_sorted[0:K,:]

    ubs = data_sorted[-K:,:]
    
    #number of features
    m = dataset.shape[1]
    
    bounds = [FloatVar(lb = [lbs[j][i] for i in range(m)] , ub=[ubs[j][i] for i in range(m)])
             for j in range(K)]
    
    return bounds






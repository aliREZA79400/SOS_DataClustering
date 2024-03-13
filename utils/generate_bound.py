import numpy as np
import data
from mealpy.utils.space import FloatVar
import validators

dataset , target = data.retu_dataset()

data_sorted = np.sort(dataset, axis=0)

def genearte(K:int):
    '''
    Args:
    K = number of clusters
    '''

    K = validators(int)

    lbs = data_sorted[0:K,:]

    ubs = data_sorted[-K:,:]
    
    #number of features
    m = dataset.shape[1]
    
    bounds = [FloatVar(lb = [lbs[j][i] for i in range(m)] , ub=[ubs[j][i] for i in range(m)])
             for j in range(K)]
    
    return bounds , dataset , target






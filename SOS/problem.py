from typing import List, Tuple
from mealpy import Problem
from mealpy.utils.space import BaseVar
import numpy as np
from numpy import ndarray
#########################################################
import sys
import os

# Get the parent directory
parent_dir = os.path.dirname(os.path.realpath("SOS"))

# Add the parent directory to sys.path
sys.path.append(parent_dir)
############################################################
from utils import validators


class Data_Clustering(Problem):

    name = validators.ValidType(str)
    K = validators.ValidType(int)

    def __init__(self, bounds: List | Tuple | ndarray | BaseVar,
                 K : int ,
                 name : str = "data_clustering",
                 dataset = None,
                 minmax: str = "min",
                 **kwargs) -> None:
        '''
        Args :
        K = number of cluster centers
        dataset  = ndarray (without labels)

        Returns :
        value of objective function
        '''
        
        self.name = name

        self.K = K

        self.dataset = dataset

        super().__init__(bounds, minmax, **kwargs)
    
    def obj_func(self, solution : ndarray) -> float:
        # solution == organism == individual population

        #number of features in dataset
        m = self.dataset.shape[1]

        norms = [[np.linalg.norm(row  - cluster_center) for cluster_center in np.reshape(np.array(solution),(self.K,m)) ] for row in self.dataset]

        valu_obj =  np.sum(

            np.fromiter((np.min(norms[i]) for i in range(len(norms))),dtype=np.float32)
        )
        
        return valu_obj
    


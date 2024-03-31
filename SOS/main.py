import mealpy 
import numpy as np
############################################################
import os
import sys
# Get the parent directory
parent_dir = os.path.dirname(os.path.realpath("SOS"))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

# Get the parent directory
parent_dir_ = os.path.dirname(parent_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir_)
print(parent_dir_)
############################################################
from utils import datasets , generate_bound , visualize
from SOS.problem import Data_Clustering
from mealpy.bio_based import SOS

K = 3

### define datatset and target

dataset , target = datasets.retu_dataset()

### generate bound

bound = generate_bound.generate(K=K,dataset=dataset)

### define problem with class {log training process}

problem_ins = Data_Clustering(bounds=bound,
                              K=K,
                              dataset=dataset)

### define model parameters dict 


### build model (instance)

model = SOS.OriginalSOS(epoch=200,pop_size=5)


### train model  with solve ( training modes )

g_best  = model.solve(problem=problem_ins)
###Extra

### tuning (not for now)

### save model

### define termination condition (not for now)

if __name__ == "__main__":

    print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    vis = visualize.Visualize(K,dataset,target=target,g_best=g_best)
    vis.draw_clustered_2D()
    vis.draw_original_data_2D()
    vis.draw_clustered_3D()
    vis.draw_original_data_3D()
    
#parser ****




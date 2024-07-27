import os
import sys

# Get the parent directory
parent_dir = os.path.dirname(os.path.realpath("SOS"))
# Add the parent directory to sys.path
sys.path.append(parent_dir)

parent_dir_ = os.path.dirname(parent_dir)
sys.path.append(parent_dir_)

import numpy as np
from utils import datasets

def save_datasets():
    dataset_names = ("Iris_Dataset" , "Breast_Cancer" , "Balance_Scale" ,
                     "Seeds" , "Statlog" , "Contraceptive_Method_Choice" ,
                     "Haberman_s_Survival", "Wine")
    
    for dataset_name in dataset_names:
        with open(dataset_name, "wb") as f:
            X ,y = datasets.retu_dataset(name=dataset_name)
            np.save(f,X)
            np.save(f,y)

        print(f"{dataset_name} saved")


if __name__ == "__main__":
    save_datasets()
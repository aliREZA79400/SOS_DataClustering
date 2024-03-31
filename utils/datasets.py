from sklearn import datasets
import os
import sys
# Get the parent directory
parent_dir = os.path.dirname(os.path.realpath("utils"))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from utils import validators

def retu_dataset(name:str = "iris_datset"):
    '''
    This function returns available datasets

    Args :
    name = name of dataset that exist in available dataset list

    Returns :
    dataset and target

    available datasets : [
    iris_dataset
    ]
    '''

    validators.validate_str(arg_name="dataset name",arg_value=name)

    if name == "iris_dataset" or "iris":

        return datasets.load_iris(return_X_y=True)
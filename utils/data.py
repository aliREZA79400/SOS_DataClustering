from sklearn import datasets
import validators


def retu_dataset(name:str = "iris_datset"):
    '''
    This function returns available datasets

    Args :
    name = name of dataset that exist in available dataset list

    available datasets : [
    iris_dataset
    ]
    '''

    name = validators.ValidType(str)

    if name == "iris_dataset" or "iris":

        dataset , target = datasets.load_iris(return_X_y=True,target = True)

        return dataset , target

from sklearn import datasets
import os
import sys
import pandas as pd

from ucimlrepo import fetch_ucirepo 

# Get the parent directory
parent_dir = os.path.dirname(os.path.realpath("utils"))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from utils import validators

def retu_dataset(name:str):
    '''
    This function returns available datasets

    Args :
    name = name of dataset that exist in available dataset list

    Returns :
    dataset and target

    available datasets : [
    Iris_Dataset , Breast_Cancer , Balance_Scale , Seeds , Statlog , Contraceptive_Method_Choice , Haberman_s_Survival , Wine
    ]
    '''
    validators.validate_str(arg_name="dataset name",arg_value=name)
    
    match name :
        case "Iris_Dataset":
            print("Iris")
            return datasets.load_iris(return_X_y=True)
        
        case "Breast_Cancer":
            return datasets.load_breast_cancer(return_X_y=True)
        
        case "Balance_Scale":
            # fetch dataset 
            balance_scale = fetch_ucirepo(id=12) 
            
            # data (as pandas dataframes) 
            X = balance_scale.data.features.to_numpy() 
            y = balance_scale.data.targets.to_numpy() 

            return X , y

        case "Seeds":
            data_path = "/media/alireza/SSD/arshad_hosh/ProCode/ADCSOS/utils/seeds_dataset.txt"

        # Read the text file (assuming it's named "seeds_dataset.txt")
            with open(data_path  ,"r") as file:
                lines = file.readlines()

            # Remove line 8 (index 7) from the list of lines (uncorrect data)
            del lines[7]

            # Create a list of dictionaries to store the data
            data = []
            for line in lines:
                fields = line.strip().split()
                data.append({
                    "Area": float(fields[0]),
                    "Perimeter": float(fields[1]),
                    "Compactness": float(fields[2]),
                    "Length of Kernel": float(fields[3]),
                    "Width of Kernel": float(fields[4]),
                    "Asymmetry Coefficient": float(fields[5]),
                    "Length of Kernel Groove": float(fields[6]),
                    "Class": int(fields[7])  # Assuming the last field is the class label
                })

            # Create a pandas DataFrame
            df = pd.DataFrame(data)

            return df.drop("Class", axis=1).to_numpy()  , df["Class"].to_numpy() 
        
        case "Statlog":
            # fetch dataset 
            statlog_heart = fetch_ucirepo(id=145) 
            
            # data (as pandas dataframes) 
            X = statlog_heart.data.features.to_numpy() 
            y = statlog_heart.data.targets.to_numpy() 

            return X , y
        
        case "Contraceptive_Method_Choice":
            # fetch dataset 
            contraceptive_method_choice = fetch_ucirepo(id=30) 
            
            # data (as pandas dataframes) 
            X = contraceptive_method_choice.data.features.to_numpy()  
            y = contraceptive_method_choice.data.targets.to_numpy() 
            return X ,y 
        
        case "Haberman_s_Survival":
            # fetch dataset 
            haberman_s_survival = fetch_ucirepo(id=43) 
            
            # data (as pandas dataframes) 
            X = haberman_s_survival.data.features.to_numpy()  
            y = haberman_s_survival.data.targets.to_numpy()
            return X , y
        
        case "Wine":
            print("this wine dataset")
            return  datasets.load_wine(return_X_y=True)

    print("dataset name is invalid")
from mealpy.utils import io
import os
import sys
# Get the parent directory
parent_dir = os.path.dirname(os.path.realpath("utils"))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from utils.validators import ValidType

def saver_model(model,save_path:str):
    """
    Args:
        model : model
        
        save_path (str)
    """
    save_path = ValidType(str)
    io.save_model(model=model,path_save=save_path)

def loader_model(loaded_path:str):
    """
    Args:
        loaded_path

    Returns:
        Trained model
    """
    loaded_path = ValidType(str)
    return io.load_model(loaded_path)



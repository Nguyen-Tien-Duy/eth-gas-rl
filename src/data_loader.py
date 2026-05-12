import pandas as pd 
import numpy as np 

def load_data(dir: str) -> dict[str, np.ndarray]:
    """
    This function loads the data from the given directory.
    The data is expected to be in CSV format.
    
    Returns:
        dict[str, np.ndarray]: A dictionary containing the data.
    """
    print(f"Loading data from {dir}..")\
    
    cols = ['base_fee', ]
    df = pd.read_csv(dir)
    return df 
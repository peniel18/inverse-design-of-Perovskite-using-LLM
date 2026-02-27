import gzip 
import random
from sklearn.model_selection import train_test_split    
import pickle
from configs.data_paths import DataPaths
from utils.logger import logger


def split_data(data, random_state=42, test_size=0.1):
    """Split data into train and test sets
    
    Args: 
        data (list): List of data samples
        random_state (int): Random seed for reproducibility
        test_size (float): Proportion of data to be used as test set
    """


    train_data, test_data = train_test_split(
        data, test_size=test_size, 
        random_state=random_state, shuffle=True
        
    )
    train_data, val_data = train_test_split(
        train_data, test_size=test_size, 
        random_state=random_state, shuffle=True
    )

    logger.info(f"Total samples: {len(data)}")
    logger.info(f"Train samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(val_data)}")
    logger.info(f"Test samples: {len(test_data)}")


    with gzip.open(DataPaths.train_data_file, "wb") as f:
        pickle.dump(train_data, f)
    
    with gzip.open(DataPaths.val_data_file, "wb") as f:
        pickle.dump(val_data, f)

    with gzip.open(DataPaths.test_data_file, "wb") as f:
        pickle.dump(test_data, f)

    return




if __name__ == "__main__":
    paths = DataPaths()

    with gzip.open(paths.processed_cifs_file, "rb") as f:
        data = pickle.load(f)

    split_data(data)

    print("Data Splitted Sucessfully")

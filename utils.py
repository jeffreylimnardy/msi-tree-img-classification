import glob
import os


def create_search_run():
    """
    Function to save the Grid Search results.
    """
    num_search_dirs = len(glob.glob('./outputs/search_*'))
    search_dirs = f"./outputs/search_{num_search_dirs+1}"
    os.makedirs(search_dirs)
    return search_dirs


def save_best_hyperparam(text, path):
    """
    Function to save best hyperparameters in a `.yml` file.
    :param text: The hyperparameters dictionary.
    :param path: Path to save the hyperparmeters.
    """
    with open(path, 'a') as f:
        f.write(f"{str(text)}\n")

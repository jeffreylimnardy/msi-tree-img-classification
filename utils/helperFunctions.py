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


def make_weights_for_balanced_classes(images, nclasses):
    n_images = len(images)
    count_per_class = [0] * nclasses
    for _, image_class in images:
        count_per_class[image_class] += 1
    weight_per_class = [0.] * nclasses
    for i in range(nclasses):
        weight_per_class[i] = float(n_images) / float(count_per_class[i])
    weights = [0] * n_images
    for idx, (image, image_class) in enumerate(images):
        weights[idx] = weight_per_class[image_class]
    return weights

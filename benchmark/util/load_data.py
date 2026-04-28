import numpy as np
import os
from pathlib import Path

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

VALID_DATA_NAMES = [
    "adult",
    "adult-race",
    "german",
    "titanic",
    "heritage-health",
    "stroke",
    "stroke-age",
    "balanced-stroke",
    "balanced-stroke-age",
    "heart",
    "heart-age",
    "balanced-heart",
    "balanced-heart-age",
    "arrhythmia",
    "arrhythmia-age",
    "balanced-arrhythmia",
    "balanced-arrhythmia-age",
]
VALID_FOLDER_NAMES = {
    "adult": "adult",
    "adult-race": "adult",
    "german": "german",
    "titanic": "titanic",
    "heritage-health": "heritage-health",
    "stroke": "stroke",
    "stroke-age": "stroke",
    "balanced-stroke": "stroke",
    "balanced-stroke-age": "stroke",
    "heart": "heart",
    "heart-age": "heart",
    "balanced-heart": "heart",
    "balanced-heart-age": "heart",
    "arrhythmia": "arrhythmia",
    "arrhythmia-age": "arrhythmia",
    "balanced-arrhythmia": "arrhythmia",
    "balanced-arrhythmia-age": "arrhythmia",
}
VALID_FILE_NAMES = {
    "adult": "adult",
    "adult-race": "adult",
    "german": "german",
    "titanic": "titanic",
    "heritage-health": "heritage-health",
    "stroke": "stroke",
    "stroke-age": "stroke",
    "balanced-stroke": "balanced_stroke",
    "balanced-stroke-age": "balanced_stroke",
    "heart": "heart",
    "heart-age": "heart",
    "balanced-heart": "balanced_heart",
    "balanced-heart-age": "balanced_heart",
    "arrhythmia": "arrhythmia",
    "arrhythmia-age": "arrhythmia",
    "balanced-arrhythmia": "balanced_arrhythmia",
    "balanced-arrhythmia-age": "balanced_arrhythmia",
}
VALID_LEARNING_STEPS = ["train", "valid", "test"]
ACCESS_INDEXES = {
    "adult": [slice(-1), -1, -2],  # [X, Y, A]
    "adult-race": [slice(-1), -1, slice(63, 68)],
    "german": [slice(-1), -1, -2],
    "titanic": [slice(-1), -1, -2],
    "heritage-health": [],
    "stroke": [slice(-1), -1, 0],
    "stroke-age": [slice(-1), -1, 1],
    "balanced-stroke": [slice(-1), -1, 0],
    "balanced-stroke-age": [slice(-1), -1, 1],
    "heart": [slice(-1), -1, 1],
    "heart-age": [slice(-1), -1, 0],
    "balanced-heart": [slice(-1), -1, 1],
    "balanced-heart-age": [slice(-1), -1, 0],
    "arrhythmia": [slice(-1), -1, 1],
    "arrhythmia-age": [slice(-1), -1, 0],
    "balanced-arrhythmia": [slice(-1), -1, 1],
    "balanced-arrhythmia-age": [slice(-1), -1, 0],
}
DIMENSIONS = {
    "adult": [112, 1, 1],  # [X, Y, A]
    "adult-race": [112, 1, 5],
    "german": [31, 1, 1],
    "titanic": [19, 1, 1],
    "heritage-health": "heritage-health",
    "stroke": [17, 1, 1],
    "stroke-age": [17, 1, 1],
    "balanced-stroke": [17, 1, 1],
    "balanced-stroke-age": [17, 1, 1],
    "heart": [26, 1, 1],
    "heart-age": [26, 1, 1],
    "balanced-heart": [26, 1, 1],
    "balanced-heart-age": [26, 1, 1],
    "arrhythmia": [278, 1, 1],
    "arrhythmia-age": [278, 1, 1],
    "balanced-arrhythmia": [278, 1, 1],
    "balanced-arrhythmia-age": [278, 1, 1],
}


def load_data(data_name, learning_step=None, kind="np"):
    """Function to load data.

    Args:
        data_name (str): used to select the correct data file.
        learning_step (str): used to select the correct data for the learning step.

    Returns:
        [type]: [description]
    """
    if not data_name in VALID_DATA_NAMES:
        print(
            "Invalid data name! Input: {} | Valid data names: [{}]",
            format(VALID_DATA_NAMES),
        )
        return None

    if learning_step is None:
        learning_step = VALID_FILE_NAMES[data_name]

    elif not learning_step in VALID_LEARNING_STEPS:
        print(
            "Invalid data name! Input: {} | Valid steps: [{}]",
            format(VALID_LEARNING_STEPS),
        )
        return None

    data_folder = select_data_folder(data_name)
    access_indexes = get_access_indexes(data_name)

    if kind == "np":
        # x, y, a = select_data_step_np(learning_step, access_indexes, data_folder, data_name)
        return select_data_step_np(
            learning_step, access_indexes, data_folder, data_name
        )
    elif kind == "pd":
        # x, y, a = select_data_step_pd(learning_step, access_indexes, data_folder, data_name)
        return select_data_step_pd(
            learning_step, access_indexes, data_folder, data_name
        )


def select_data_folder(data_name):
    return os.path.join(
        ROOT_DIR, Path(r"../data/{}".format(VALID_FOLDER_NAMES[data_name]))
    )


def get_access_indexes(data_name):
    return ACCESS_INDEXES[data_name]


def select_data_step_np(learning_step, access_indexes, data_folder, data_name):
    file = os.path.join(data_folder, Path(r"post_prep/{}.csv".format(learning_step)))
    data = np.genfromtxt(file, delimiter=",", skip_header=True)[:, 1:]

    num_examples = data.shape[0]
    x = data[:, access_indexes[0]]
    y = data[:, access_indexes[1]].reshape(num_examples, DIMENSIONS[data_name][1])
    a = data[:, access_indexes[2]].reshape(num_examples, DIMENSIONS[data_name][2])

    return x, y, a


def select_data_step_pd(learning_step, access_indexes, data_folder, data_name):
    file = os.path.join(data_folder, Path(r"post_prep/{}.csv".format(learning_step)))

    data = pd.read_csv(file)

    return data

from math import sqrt, isnan

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from util import metrics

from fairgan.models import *

def train(model, X, Y, A, optimizer, alpha=1):
    pass

def train_loop(model, raw_data, train_dataset, epochs, opt=None):
    
    print("> Epoch | Class Loss | Adv Loss | Class Acc | Adv Acc")

    pass
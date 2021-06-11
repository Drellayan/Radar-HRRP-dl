import torch
from torch import nn
from d2l import torch as d2l
import math

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import scipy.io as scio
import numpy as np


import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

loss = nn.CrossEntropyLoss()
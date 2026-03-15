import json, math, os, random, time, textwrap, zipfile
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from reportlab.platypus import SimpleDocTemplate

print("All libraries imported successfully!")

import keras
import numpy as np

from pathlib import Path
from models import BestSCNN
from MEGDataset import MEGAgeRangesDataset

TRAINED_MODEL = Path('/ais/clspace5/spoclab/children_megdata/eeglabdatasets/models/MEGraw/scnn/bin/')

if __name__ == '__main__':

    dataset = MEGAgeRangesDataset('/ais/clspace5/spoclab/children_megdata/biggerset')



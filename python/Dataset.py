import pickle
import numpy as np
import mne



def zscore(l, nanprevention=1e-20):
    l = np.float32(l)
    return (l - np.mean(l, axis=0)) / (np.std(l, axis=0) + nanprevention)




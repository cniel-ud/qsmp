import numpy as np


def _gauss(sublen, support_frac):
    sigma = support_frac * sublen / 6
    t = np.arange(sublen)[None, :]
    win = np.exp(-0.5*((t-sublen/2)/sigma)**2)
    return win

def _rect(sublen, support_frac):
    support = int(support_frac * sublen)
    win = np.zeros(sublen)
    idx = np.arange((sublen-support)//2, (sublen+support)//2)
    win[idx] = 1.0
    return win[None, :]

WINDOWS = {
    'gauss': _gauss,
    'rect': _rect
}

def get_window(window_type):
    return WINDOWS[window_type]

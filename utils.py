import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle




def make_cmap(colouring:dict, sqn:int, cmap_str:str = 'Spectral'):
    digits = sqn**2
    if sqn in known_cmaps.keys(): cmap_str = known_cmaps[sqn]
    cmap = cm.get_cmap(cmap_str)
    maxcol = max(colouring.values())
    return [cmap(k/(digits-1)) for k in range(digits)] + [(0,0,0,0.4)]*int(abs(maxcol+1-digits)), cmap_str
    
    
    
s3_2 = LinearSegmentedColormap('s3_2', {
    'red':[(0.0, 0.0, 0.0),
           (0.25,0.1, 0.1),
           (0.5, 0.85, 0.85),
           (1.0,1.0, 1.0)],
    'green':[(0.0, 1.0, 1.0),
             (0.25,0.5, 0.5),
             (0.75, 0.0, 0.0),
             (1.0,0.5, 0.5)],
    'blue':[(0.0, 0.5, 0.5),
            (0.25,0.75, 0.75),
            (0.75, 0.9, 0.9),
            (1.0,0.1, 0.1)]    
})
plt.register_cmap(cmap=s3_2)

known_cmaps = {2:'s3_2', 3:'Set1'}
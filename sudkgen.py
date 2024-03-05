import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from utils import make_cmap
#import nxviz as nv
import sudktools as sudk










if __name__ == '__main__':
    N = 4
    sqn = 3
    digits = sqn**2
    
    wireframe_on = True
    voxels_on = True
    voxel_projection_on = False
    save = False
    custom_handling = True
    display = True
    
    S = sudk.Sudoku(digits=digits,N=N)
    G = nx.from_numpy_array(S.adjacency)
    subG = nx.from_numpy_array(S.subcube_adjacency)
    
    # color the sqn**N subgraph
    S.subcube_solve(range(sqn**N)) # nx.coloring.strategy_largest_first(subG,{})
    Hsub = {node:S.subcube[S.subcube_idx(node)]-1 for node in range(sqn**N)}
    print(Hsub)
    print(S.subcube)
    #exit()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for e in subG.edges():
        E = np.array([S.subcube_idx(e[0]), S.subcube_idx(e[1])])
        ax.plot(E[:,0],E[:,1],E[:,2], color=(0.5,0.5,0.5,0.4))    
    Csub, csub_str = make_cmap(Hsub, sqn)
    colouring = np.array([Csub[Hsub[v]] for v in subG.nodes()])
    POS = np.array([S.subcube_idx(n) for n in subG.nodes()])
    if voxels_on:
        for i,p in enumerate(POS):
            vox = np.zeros((sqn,sqn,sqn)); vox[p[0],p[1],p[2]] = 1
            ax.voxels(*(np.indices((sqn+1,sqn+1,sqn+1))-1/2), vox, color=tuple(colouring[i][:3])+(0.15,))
            ax.text(p[0],p[1],p[2],f'{Hsub[i]+1}', size=10)
    plt.show()
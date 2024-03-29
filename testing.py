import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import networkx as nx
from utils import make_cmap
#import nxviz as nv
from skimage import io
import sudktools as sudk


def K(n1, n2=None):
    if n2 == None:
        return np.ones((n1,n1)) - np.diag(np.ones(n1))
    G = np.ones((n1+n2,n1+n2)); G[:n1,:n1], G[n1:,n1:] = 0,0
    return G

def orthogonal(n):
    sqn = int(np.sqrt(n))
    G = np.zeros((n,n))
    indices = np.array([i*sqn for i in range(sqn)])
    for k in range(sqn): G[indices+k, sqn*k:sqn*(k+1)]=1
    return G





if __name__ == '__main__':
    N = 3
    sqn = 2
    digits = sqn**2
    
    wireframe_on = True
    voxels_on = True
    voxel_projection_on = False
    save = False
    custom_handling = True
    display = True
    
    S = sudk.Sudoku(digits=digits,N=N,constraints=())
    G = nx.from_numpy_array(S.adjacency)
    subcube = nx.from_numpy_array(S.subcube)
    
    
    
    
    
    
    init_strat = None
    if init_strat is None:
        D = {}
        for strat in ('saturation_largest_first','independent_set','largest_first'):
            H = nx.greedy_color(G, strat)
            mH = max(H.values())+1
            name = strat if type(strat) == str else strat.__name__
            print(f'number of colours {name} : ', mH)
            D[name] = (H,mH,strat)
        results = list(D.values()); results = [(i,res[1]) for i,res in enumerate(results)]
        results = np.sort(np.array(results, dtype=[('x',int),('y',int)]),order='y')
        idx = results[0][0]
        H, mH, strat = list(D.values())[idx]
        print(f'Best : {strat} with {mH} colours.')
    else:
        H = nx.greedy_color(G, init_strat)
        mH = max(H.values())+1
        name = init_strat if type(init_strat) == str else init_strat.__name__
        print(f'number of colours {name} : ', mH)

    
    
    C, c_str = make_cmap(H, sqn)
    
    if not display: exit()

    if N == 2:
        fig = plt.figure()
        fig.set_figheight(4.8), fig.set_figwidth(6)
        
        ax = fig.add_subplot(111)
        colouring = np.array([C[H[v]] for v in G.nodes()])
        POS = np.array([S.find_indices(n) for n in G.nodes()])
        ax.scatter(POS[:,0],POS[:,1], color=(0,0,0,0))
        if voxel_projection_on or voxels_on:
            for i, p in enumerate(POS): 
                ax.add_patch(Rectangle(
                    xy=(p[0]-1/2, p[1]-1/2) ,width=1, height=1,
                    linewidth=1, color=tuple(colouring[i][:3])+(0.4,)))
                ax.text(p[0]-0.1, p[1]-0.1, f'{H[i]+1}')
        if save:
            img = np.zeros((digits,)*N)
            for i,p in enumerate(POS):
                img[p[0],p[1],p[2]] = H[i]+1
            io.imsave(f'N3_digits{digits}.tif',img)
        if wireframe_on:
            for e in G.edges():
                E = np.array([S.find_indices(e[0]), S.find_indices(e[1])])
                ax.plot(E[:,0],E[:,1], color=(0.5,0.5,0.5,0.4))
        ax.set_xticks([]),ax.set_yticks([])
        #fig.suptitle('9x9 Sudoku colouring attempt with King & Knight constraints')
        #nx.draw(G, with_labels=True, font_weight='bold', pos=dict( (n,S.find_indices(n)) for n in G.nodes() ))
    elif N == 3: 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        
        colouring = np.array([C[H[v]] for v in G.nodes()])
        POS = np.array([S.find_indices(n) for n in G.nodes()])
        if voxel_projection_on:
            ax.scatter(POS[:,0],POS[:,1],POS[:,2], color=colouring, marker='s', s=300)        
        if voxels_on:
            for i,p in enumerate(POS):
                vox = np.zeros((digits,digits,digits)); vox[p[0],p[1],p[2]] = 1
                ax.voxels(*(np.indices((digits+1,digits+1,digits+1))-1/2), vox, color=tuple(colouring[i][:3])+(0.15,))
                ax.text(p[0],p[1],p[2],f'{H[i]+1}', size=10)
        if save:
            img = np.zeros((digits,)*N)
            for i,p in enumerate(POS):
                img[p[0],p[1],p[2]] = H[i]+1
            io.imsave(f'N3_digits{digits}.tif',img)
        if wireframe_on:
            for e in G.edges():
                E = np.array([S.find_indices(e[0]), S.find_indices(e[1])])
                ax.plot(E[:,0],E[:,1],E[:,2], color=(0.5,0.5,0.5,0.4))
        ax.set_xticks([]),ax.set_yticks([]),ax.set_zticks([])
    elif N == 4 and sqn == 2 and custom_handling: 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        
        colouring = np.array([C[H[v]] for v in G.nodes()])
        POS = np.array([S.find_indices(n) for n in G.nodes()])
        
        tetra = lambda x: np.array([0,0,0]) if x<1 else np.array([1,0,0]) if x<2 else np.array([0.5,0.87,0]) if x<3 else np.array([0.5,0.29,0.82])
        
        if voxel_projection_on:
            ax.scatter(*np.array(np.array([POS[:,0],POS[:,1],POS[:,2]]) + 5*np.array([tetra(d4) for d4 in POS[:,3]]).transpose(1,0)), color=colouring, marker='s', s=300)        
            #for i,p in enumerate(POS):
            #    ax.text(p[0]+5*(1*(p[3]>0)-0.5*(p[3]>1)),p[1]+5*(0.87*(p[3]>1)-0.58*(p[3]>2)),p[2]+5*(0.82*(p[3]>2)),f'{H[i]+1}', size=10)
                
        if wireframe_on:
            for e in G.edges():
                E = np.array([S.find_indices(e[0]), S.find_indices(e[1])])
                ax.plot(*np.array(np.array([E[:,0],E[:,1],E[:,2]]) + 5*np.array([tetra(d4) for d4 in E[:,3]]).transpose(1,0)), color=(0.5,0.5,0.5,0.4))
        ax.set_xticks([]),ax.set_yticks([]),ax.set_zticks([])
    else:
        colouring = np.array([C[H[v]] for v in G.nodes()])
        POS = np.array([S.find_indices(n) for n in G.nodes()])
        img = np.zeros((digits,)*N)
        for i,p in enumerate(POS):
            img[tuple(p)] = H[i]+1
        io.imsave(f'N{N}_digits{digits}.tif',img)
        # use the line below in case of wrongful RGB/RGBA detection
        # viewer.add_image(viewer.layers[-1].data, rgb=False)
        
    if (N < 4 or (N,sqn)==(4,2)) and custom_handling:
        cbar = fig.colorbar(cm.ScalarMappable(cmap=c_str), ax=ax, ticks=np.linspace(0,1,digits))
        cbar.ax.set_yticklabels(range(1,digits+1))
        plt.show()









import numpy as np
from copy import deepcopy
import pdb

def N_choose_two(N):
    elements = [i for i in range(N)]
    projections = []
    while elements != []:
        for n in elements[1:]:
            projections.append(tuple([i for i in range(N) if i not in (elements[0], n)]) + (elements[0],n))
        elements.pop(0)
    return projections

class Sudoku:
    def __init__(self,N=2,digits=9, **kwargs):
        if N < 2: N = 2
        self.N = N
        if digits != int(np.sqrt(digits))**2 : raise ValueError('digits must be a perfect square')
        self.digits = digits
        self.shape = tuple([digits]*N)
        self.grid = np.zeros(self.shape, dtype=np.int8)
        self.numbers = np.array([n+1 for n in range(digits)])
        self.adjacency = np.zeros((self.grid.size, self.grid.size))
        if 'constraints' in kwargs.keys():
            self.xtra_rules = kwargs['constraints']
        self.fill_adj()
    
    def find_indices(self, pos):
        """Find tuple of indices corresponding to integer positive in ndarray.
            
        Example:
            ```py
            >>> sudk = Sudoku(N=4)
            >>> grid = sudk.grid
            >>> sudk.find_indices(5420)
            (7,3,8,3) # gives slice position of 5420th element
            >>> grid[(7,3,8,3)] # how we access the element
            ```
            This allows us to loop on a range of `grid.size` and associate
            each element to a specific cell.

        """
        d = self.digits
        indices = []
        for i in range(self.N-1,-1,-1):
            index = pos//(d**i)
            pos -= index*d**i
            indices.append(index)
        return tuple(indices)
    
    def in_orthogonal(self, grid, indices):
        """Check if a number is already present on the lines of a cell, given its position.
        The word "lines" is used regardless of the axis considered, as the distinction 
        between line and column is meaningless in higher dimensions.

        Returns:
            _type_: _description_
        """
        G = deepcopy(grid)
        axes = [ax for ax in range(self.N)]
        for _ in range(self.N):
            projected_idx = tuple([indices[i] for i in axes])
            G.transpose(tuple(axes))[projected_idx[:-1]] = 1
            axes.append(axes.pop(0))
        return G
    
    def in_square(self, grid, indices): # tN different projections (because N choose 2 dimensions to project on)
        G = deepcopy(grid)
        sq_size = int(np.sqrt(self.digits))
        G = G != G
        for projection in N_choose_two(self.N):
            # [[indices[i] for i in projection][:-2]]
            # permutation of indices using projection
            # [indices[:-2]]
            # otherwise
            subgrid = G.transpose(projection)[tuple([indices[i] for i in projection][:-2])]
            line, column = [indices[i] for i in projection][-2:]
            # subline = subgrid[:3] if line < 3 else subgrid[3:6] if line < 6 else subgrid[6:]
            # square = subline[:,:3] if column < 3 else subline[:,3:6] if column < 6 else subline[:,6:]
            # generalising lines above by using evaluation of reg string
            subline = eval(f'(0,{sq_size})'+\
                ''.join([f' if line < {k-sq_size} else ({k-sq_size},{k})' for k in range(sq_size*2, self.digits+1, sq_size)]))
            square = eval(f'(0,{sq_size})'+\
                ''.join([f' if column < {k-sq_size} else ({k-sq_size},{k})' for k in range(sq_size*2, self.digits+1, sq_size)]))            
            subgrid[subline[0]:subline[1],square[0]:square[1]] = 1
        return G
            
    def breaks_constraints(self, grid, indices):
        """Add constraints to the grid generation.

        Current constraints accepted are 'knight' and 'king', can't be used together and only work for N=2 and digits=9.
        """
        # constraints only for N = 2 and digits = 9 because of bad generalisations
        if self.xtra_rules == [] or self.N != 2 or self.digits != 9: return False

        G = deepcopy(grid)
        if 'king' in self.xtra_rules:
            line, column = indices
            T,B,L,R = line>0, line<grid.shape[0]-1, column>0, column<grid.shape[1]-1
            neighbours_idx = [idx for idx in
                        [(line-1,column-1)]*T*L + [(line-1,column)]*T + [(line-1,column+1)]*T*R + \
                        [(line,column-1)]*L     +                             [(line,column+1)]*R + \
                        [(line+1,column-1)]*B*L + [(line+1,column)]*B + [(line+1,column+1)]*B*R]
            for idx in neighbours_idx: G[idx] = 1

        if 'knight' in self.xtra_rules:
            line, column = indices
            
            neighbour_positions = np.array([(line+m,column+n) for m in range(-2,3) for n in range(-2,3) if abs(m)+abs(n) == 3])
            nei_in_grid = (neighbour_positions.transpose(1,0)[0] > 0) * (neighbour_positions.transpose(1,0)[0] < G.shape[0]-1) * \
                          (neighbour_positions.transpose(1,0)[1] > 0) * (neighbour_positions.transpose(1,0)[1] < G.shape[1])
            # checks if knight positions are in grid
            for idx in [tuple(neighbour_positions[i]) for i in range(neighbour_positions.shape[0]) if nei_in_grid[i]]:
                G[idx] = 1
        
        
        return G

                
    def fill_adj(self):
        """Fill an empty grid conformly to how sudokus rules work.

        Returns:
            _type_: _description_
        """
        for pos in range(self.grid.size):
            connected = np.zeros(self.shape)
            indices = self.find_indices(pos)
            connected += self.in_orthogonal(self.grid,indices) + self.in_square(self.grid,indices) + self.breaks_constraints(self.grid, indices)
            connected = np.reshape(connected, (1,self.grid.size))[0]
            self.adjacency[pos] = connected>0
        self.adjacency -= np.diag(np.ones(self.grid.size))
        


if __name__ == '__main__':

    sudoku = Sudoku(digits=9,N=3)
    print(sudoku.grid)
    print(sudoku.adjacency)

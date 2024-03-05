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

def get_perm_of_orbit_length(k): return lambda iterable : np.array([iterable[j+1] for j in range(k-1)] + [iterable[0]])

class SolvabilityError(BaseException):
    """Raise when sudoku isn't solvable for some specified reason.
    """

class Sudoku:
    def __init__(self,N=2,digits=9, **kwargs):
        if N < 2: N = 2
        self.N = N
        self.sqn = int(np.sqrt(digits))
        if digits != self.sqn**2 : raise ValueError('digits must be a perfect square')
        if N > self.sqn+1: raise SolvabilityError('Number of dimensions N should be less or equal to sqn+1')
        self.digits = digits
        self.shape = tuple([digits]*N)
        self.grid = np.zeros(self.shape, dtype=np.int8)
        self.numbers = np.array([n+1 for n in range(digits)])
        self.adjacency = np.zeros((self.grid.size, self.grid.size))
        self.subcube = np.zeros((self.sqn,)*N,int)
        self.subcube_adjacency = np.zeros((self.sqn**N,self.sqn**N))
        self.xtra_rules = ()
        if 'constraints' in kwargs.keys(): self.xtra_rules = kwargs['constraints']
        self.fill_adj()
        self.subcube_adj()
    
    def find_indices(self, pos):
        """Find tuple of indices corresponding to integer positive in ndarray.
            
        Example:
            
            >>> sudk = Sudoku(N=4)
            >>> grid = sudk.grid
            >>> sudk.find_indices(5420)
            (7,3,8,2) # gives slice position of 5420th element
            >>> grid[(7,3,8,2)] # how we access the element
            
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
    
    def in_orthogonal_adj(self, grid, indices):
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
    
    def in_square_adj(self, grid, indices): # tN different projections (because N choose 2 dimensions to project on)
        G = deepcopy(grid)
        sq_size = int(np.sqrt(self.digits))
        G = G != G
        for e,projection in enumerate(N_choose_two(self.N)):
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
            
    def breaks_constraints_adj(self, grid, indices):
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

    
    def in_orthogonal(self, grid, indices, n):
        """Check if a number is already present on the lines of a cell, given its position.
        The word "lines" is used regardless of the axis considered, as the distinction 
        between line and column is meaningless in higher dimensions.

        Returns:
            _type_: _description_
        """
        on_line = False
        axes = [ax for ax in range(self.N)]
        for _ in range(self.N):
            projected_idx = tuple([indices[i] for i in axes])
            if n in grid.transpose(tuple(axes))[projected_idx[:-1]]: return True
            axes.append(axes.pop(0))
        return on_line
    
    def in_square(self, grid, indices, n): # tN different projections (because N choose 2 dimensions to project on)
        in_any_square = False
        sq_size = int(np.sqrt(self.digits))
        for projection in N_choose_two(self.N):
            # [[indices[i] for i in projection][:-2]]
            # permutation of indices using projection
            # [indices[:-2]]
            # otherwise
            subgrid = grid.transpose(projection)[tuple([indices[i] for i in projection][:-2])]
            line, column = [indices[i] for i in projection][-2:]
            
            # subline = subgrid[:3] if line < 3 else subgrid[3:6] if line < 6 else subgrid[6:]
            # square = subline[:,:3] if column < 3 else subline[:,3:6] if column < 6 else subline[:,6:]
            # generalising lines above by using evaluation of reg string
            subline = eval(f'subgrid[:{sq_size}]'+\
                ''.join([f' if line < {k-sq_size} else subgrid[{k-sq_size}:{k}]' for k in range(sq_size*2, self.digits+1, sq_size)]))
            square = eval(f'subline[:,:{sq_size}]'+\
                ''.join([f' if column < {k-sq_size} else subline[:,{k-sq_size}:{k}]' for k in range(sq_size*2, self.digits+1, sq_size)]))            
            if n in square: return True
        return in_any_square
            
    def breaks_constraints(self, grid, indices, n):
        """Add constraints to the grid generation.

        Current constraints accepted are 'knight' and 'king', can't be used together and only work for N=2 and digits=9.
        """
        # constraints only for N = 2 and digits = 9 because of bad generalisations
        if self.xtra_rules == [] or self.N != 2 or self.digits != 9: return False
        breaks = False

        if 'king' in self.xtra_rules:
            line, column = indices
            T,B,L,R = line>0, line<grid.shape[0]-1, column>0, column<grid.shape[1]-1
            neighbours = [grid[idx] for idx in
                        [(line-1,column-1)]*T*L + [(line-1,column)]*T + [(line-1,column+1)]*T*R + \
                        [(line,column-1)]*L     +                             [(line,column+1)]*R + \
                        [(line+1,column-1)]*B*L + [(line+1,column)]*B + [(line+1,column+1)]*B*R]
            if n in neighbours: return True

        if 'knight' in self.xtra_rules:
            line, column = indices
            neighbour_positions = np.array([(line+m,column+n) for m in range(-2,3) for n in range(-2,3) if abs(m)+abs(n) == 3])
            nei_in_grid = (neighbour_positions.transpose(1,0)[0] > 0) * (neighbour_positions.transpose(1,0)[0] < grid.shape[0]-1) * \
                          (neighbour_positions.transpose(1,0)[1] > 0) * (neighbour_positions.transpose(1,0)[1] < grid.shape[1])
            # checks if knight positions are in grid
            neighbours = [grid[idx] for idx in 
                        [tuple(neighbour_positions[i]) for i in range(neighbour_positions.shape[0]) 
                            if nei_in_grid[i]]]
            if n in neighbours: return True
        
        return breaks

                
    def fill_adj(self):
        """Fill an empty grid conformly to how sudokus rules work.

        Returns:
            _type_: _description_
        """
        for pos in range(self.grid.size):
            connected = np.zeros(self.shape)
            indices = self.find_indices(pos)
            connected += self.in_orthogonal_adj(self.grid,indices) + self.in_square_adj(self.grid,indices) + self.breaks_constraints_adj(self.grid, indices)
            self.adjacency[pos] = connected.reshape(self.grid.size)>0
        self.adjacency -= np.diag(np.ones(self.grid.size))
        
    def subcube_idx(self, pos):
            indices = []
            for i in range(self.N-1,-1,-1):
                index = pos//(self.sqn**i)
                pos -= index*self.sqn**i
                indices.append(index)
            return tuple(indices)
    
    def subcube_adj(self):
        d = int(np.sqrt(self.digits))
        
        def _subcube_orth(idx):
            G = np.zeros((d,)*self.N)
            axes = [ax for ax in range(self.N)]
            for _ in range(self.N):
                projected_idx = tuple([idx[i] for i in axes])
                G.transpose(tuple(axes))[projected_idx[:-1]] = 1
                axes.append(axes.pop(0))
            return G
        
        def _subcube_square(idx): # tN different projections (because N choose 2 dimensions to project on)
            G = np.zeros((d,)*self.N)
            for projection in N_choose_two(self.N): G.transpose(projection)[tuple([idx[i] for i in projection][:-2])] = 1
            return G
        
        def _subcube_constr(idx):
            if self.xtra_rules == [] or self.N != 2 or self.digits != 9: return False
            G = np.zeros((d,)*self.N)
            if 'king' in self.xtra_rules:
                line, column = idx
                T,B,L,R = line>0, line<G.shape[0]-1, column>0, column<G.shape[1]-1
                neighbours_idx = [idx for idx in
                            [(line-1,column-1)]*T*L + [(line-1,column)]*T + [(line-1,column+1)]*T*R + \
                            [(line,column-1)]*L     +                             [(line,column+1)]*R + \
                            [(line+1,column-1)]*B*L + [(line+1,column)]*B + [(line+1,column+1)]*B*R]
                for idx in neighbours_idx: G[idx] = 1
            if 'knight' in self.xtra_rules:
                line, column = idx
                neighbour_positions = np.array([(line+m,column+n) for m in range(-2,3) for n in range(-2,3) if abs(m)+abs(n) == 3])
                nei_in_grid = (neighbour_positions.transpose(1,0)[0] > 0) * (neighbour_positions.transpose(1,0)[0] < G.shape[0]-1) * \
                            (neighbour_positions.transpose(1,0)[1] > 0) * (neighbour_positions.transpose(1,0)[1] < G.shape[1])
                # checks if knight positions are in grid
                for idx in [tuple(neighbour_positions[i]) for i in range(neighbour_positions.shape[0]) if nei_in_grid[i]]:
                    G[idx] = 1
            return G
        
        for pos in range(d**self.N):
            idx = self.subcube_idx(pos)
            connected = _subcube_orth(idx) + _subcube_square(idx) + _subcube_constr(idx)
            self.subcube_adjacency[pos] = connected.reshape(d**self.N)>0

    def subcube_solve(self, V=None):
        if V == None: V = range(self.subcube.size)
        def _subcube_in_orth(idx, n):
            axes = [ax for ax in range(self.N)]
            for _ in range(self.N):
                projected_idx = tuple([idx[i] for i in axes])
                if n in self.subcube.transpose(tuple(axes))[projected_idx[:-1]]: return True
                axes.append(axes.pop(0))
            return False
        def _subcube_in_square(idx, n): # tN different projections (because N choose 2 dimensions to project on)
            for projection in N_choose_two(self.N):
                if n in self.subcube.transpose(projection)[tuple([idx[i] for i in projection][:-2])]: return True
            return False
        def _subcube_breaks_constraints(idx, n):
            # constraints only for N = 2 and digits = 9 because of bad generalisations
            if self.xtra_rules == [] or self.N != 2 or self.digits != 9: return False
            breaks = False

            if 'king' in self.xtra_rules:
                line, column = idx
                T,B,L,R = line>0, line<self.subcube.shape[0]-1, column>0, column<self.subcube.shape[1]-1
                neighbours = [self.subcube[idx] for idx in
                            [(line-1,column-1)]*T*L + [(line-1,column)]*T + [(line-1,column+1)]*T*R + \
                            [(line,column-1)]*L     +                             [(line,column+1)]*R + \
                            [(line+1,column-1)]*B*L + [(line+1,column)]*B + [(line+1,column+1)]*B*R]
                if n in neighbours: return True

            if 'knight' in self.xtra_rules:
                line, column = idx
                neighbour_positions = np.array([(line+m,column+n) for m in range(-2,3) for n in range(-2,3) if abs(m)+abs(n) == 3])
                nei_in_grid = (neighbour_positions.transpose(1,0)[0] > 0) * (neighbour_positions.transpose(1,0)[0] < self.subcube.shape[0]-1) * \
                            (neighbour_positions.transpose(1,0)[1] > 0) * (neighbour_positions.transpose(1,0)[1] < self.subcube.shape[1])
                # checks if knight positions are in grid
                neighbours = [self.subcube[idx] for idx in 
                            [tuple(neighbour_positions[i]) for i in range(neighbour_positions.shape[0]) 
                                if nei_in_grid[i]]]
                if n in neighbours: return True
            
            return breaks
        
        numbers = self.numbers
        for pos in V:
            idx = self.subcube_idx(pos)
            if self.subcube[idx] == 0:
                for n in numbers:
                    if not _subcube_in_orth(idx,n) and not _subcube_in_square(idx,n) and not _subcube_breaks_constraints(idx,n):
                        self.subcube[idx] = n
                        if not (0 in self.subcube): return True
                        elif self.subcube_solve(V): return True
                        self.subcube[idx] = 0
                return False
            
    
    
    def solve(self, V, thresh=1):
        numbers = self.numbers
        for pos in V:
            indices = self.find_indices(pos)
            if self.grid[indices] == 0:
                for n in numbers:
                    if not self.in_orthogonal(self.grid,indices,n) and not self.in_square(self.grid,indices,n) and not self.breaks_constraints(self.grid,indices,n):
                        self.grid[indices] = n
                        if not (0 in self.grid): return True
                        elif self.solve(V, thresh): return True
                        self.grid[indices] = 0
                return False

    def is_valid_solution(self):
        def _validity_test_orthogonal(grid, indices, n):
            grid[indices] = 0
            axes = [ax for ax in range(self.N)]
            for _ in range(self.N):
                projected_idx = tuple([indices[i] for i in axes])
                if n in grid.transpose(tuple(axes))[projected_idx[:-1]]: grid[indices] = n; return True
                axes.append(axes.pop(0))
            grid[indices] = n
            return False
        def _validity_test_square(grid, indices, n): 
            grid[indices] = 0
            sq_size = int(np.sqrt(self.digits))
            for projection in N_choose_two(self.N):
                # [[indices[i] for i in projection][:-2]]
                # permutation of indices using projection
                # [indices[:-2]]
                # otherwise
                subgrid = grid.transpose(projection)[tuple([indices[i] for i in projection][:-2])]
                line, column = [indices[i] for i in projection][-2:]
                
                # subline = subgrid[:3] if line < 3 else subgrid[3:6] if line < 6 else subgrid[6:]
                # square = subline[:,:3] if column < 3 else subline[:,3:6] if column < 6 else subline[:,6:]
                # generalising lines above by using evaluation of reg string
                subline = eval(f'subgrid[:{sq_size}]'+\
                    ''.join([f' if line < {k-sq_size} else subgrid[{k-sq_size}:{k}]' for k in range(sq_size*2, self.digits+1, sq_size)]))
                square = eval(f'subline[:,:{sq_size}]'+\
                    ''.join([f' if column < {k-sq_size} else subline[:,{k-sq_size}:{k}]' for k in range(sq_size*2, self.digits+1, sq_size)]))            
                if n in square: grid[indices] = n; return True
            grid[indices] = n
            return False
        def _validity_test_constraints(grid, indices, n):
            # constraints only for N = 2 and digits = 9 because of bad generalisations
            if self.xtra_rules == [] or self.N != 2 or self.digits != 9: return False
            grid[indices] = 0
            if 'king' in self.xtra_rules:
                line, column = indices
                T,B,L,R = line>0, line<grid.shape[0]-1, column>0, column<grid.shape[1]-1
                neighbours = [grid[idx] for idx in
                            [(line-1,column-1)]*T*L + [(line-1,column)]*T + [(line-1,column+1)]*T*R + \
                            [(line,column-1)]*L     +                             [(line,column+1)]*R + \
                            [(line+1,column-1)]*B*L + [(line+1,column)]*B + [(line+1,column+1)]*B*R]
                if n in neighbours: grid[indices] = n; return True
            if 'knight' in self.xtra_rules:
                line, column = indices
                neighbour_positions = np.array([(line+m,column+n) for m in range(-2,3) for n in range(-2,3) if abs(m)+abs(n) == 3])
                nei_in_grid = (neighbour_positions.transpose(1,0)[0] > 0) * (neighbour_positions.transpose(1,0)[0] < grid.shape[0]-1) * \
                            (neighbour_positions.transpose(1,0)[1] > 0) * (neighbour_positions.transpose(1,0)[1] < grid.shape[1])
                # checks if knight positions are in grid
                neighbours = [grid[idx] for idx in 
                            [tuple(neighbour_positions[i]) for i in range(neighbour_positions.shape[0]) 
                                if nei_in_grid[i]]]
                if n in neighbours: grid[indices] = n; return True
            grid[indices] = n
            return False
    
        for pos in range(self.grid.size):
            indices = self.find_indices(pos)
            if _validity_test_constraints(self.grid, indices, self.grid[indices]) \
            or _validity_test_square(self.grid, indices, self.grid[indices]) \
            or _validity_test_constraints(self.grid, indices, self.grid[indices]): return False
        return True

    def solve_clever(self):
        N, n = self.N, self.sqn
        d = self.digits
        self.subcube_solve()
        sigma = get_perm_of_orbit_length(n)
        #shape = np.array([n]*N)
        #shape[0] *= n
        s = self.subcube
        for M in range(N):
            S = []
            for k in range(n):
                S.append(s)
                if M%2 == 0: s = np.array([sigma(sk.T).T for sk in s])
                else: s = np.array([sigma(sk) for sk in s])
                
            s = np.concatenate(S)
        
        s = np.concatenate(s,0)
        s = s.reshape(s.size)
        for p, v in enumerate(s): self.grid[self.find_indices(p)] = v
        
        print(self.grid)
        print(self.is_valid_solution())
        
        #print(S)
        #print(np.array(S))
        #shape = shape.transpose([k+1 for k in range(N-1)]+[0])
        #shape[0] *= n
        
        #print(np.c_[S[0],S[1]])
        


if __name__ == '__main__':

    sudoku = Sudoku(digits=9,N=2)
    #print(sudoku.grid)
    #print(sudoku.adjacency)
    #print(sudoku.grid)
    sudoku.solve_clever()
    
    

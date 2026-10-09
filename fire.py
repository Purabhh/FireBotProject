import random
import numpy as np

def init_burning(grid, cell):
    D = grid.shape[0]
    burning = np.zeros((D, D), dtype=bool)
    x, y = cell
    burning[x, y] = True
    return burning


def step_fire(grid, burning, q):
    """Advance the fire one timestep.

    grid: open mask (1=open, 0=blocked)
    burning: boolean array current burning cells
    q: flammability parameter
    returns new burning mask
    """
    D = grid.shape[0]
    new_burning = burning.copy()
    for x in range(D):
        for y in range(D):
            if grid[x, y] == 1 and not burning[x, y]:
                K = 0
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < D and 0 <= ny < D and burning[nx, ny]:
                        K += 1
                if K > 0:
                    p = 1 - (1 - q) ** K
                    if random.random() < p:
                        new_burning[x, y] = True
    return new_burning

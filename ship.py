import random
import numpy as np

def generate_ship(D, seed=None):
    """Generate a ship layout per project spec.

    Returns a numpy array of shape (D,D) with 1=open, 0=blocked.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    grid = np.zeros((D, D), dtype=np.int8)

    # choose an interior start
    i = random.randint(1, D - 2)
    j = random.randint(1, D - 2)
    grid[i, j] = 1

    def neighbors(x, y):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < D and 0 <= ny < D:
                yield nx, ny

    # Iteratively open blocked cells with exactly one open neighbor.
    while True:
        candidates = []
        for x in range(1, D - 1):
            for y in range(1, D - 1):
                if grid[x, y] == 0:
                    cnt = sum(1 for nx, ny in neighbors(x, y) if grid[nx, ny] == 1)
                    if cnt == 1:
                        candidates.append((x, y))
        if not candidates:
            break
        x, y = random.choice(candidates)
        grid[x, y] = 1

    # Identify dead ends (open cells with one open neighbor)
    def open_neighbors(x, y):
        return sum(1 for nx, ny in neighbors(x, y) if grid[nx, ny] == 1)

    dead_ends = [(x, y) for x in range(D) for y in range(D) if grid[x, y] == 1 and open_neighbors(x, y) == 1]
    target = max(0, len(dead_ends) // 2)

    # Open random closed neighbors of random dead ends until dead ends reduced by at least half
    while len(dead_ends) > target:
        de = random.choice(dead_ends)
        closed = [ (nx, ny) for nx, ny in neighbors(*de) if grid[nx, ny] == 0 ]
        if not closed:
            dead_ends.remove(de)
            continue
        nx, ny = random.choice(closed)
        grid[nx, ny] = 1
        dead_ends = [(x, y) for x in range(D) for y in range(D) if grid[x, y] == 1 and open_neighbors(x, y) == 1]

    return grid


def get_open_cells(grid):
    D = grid.shape[0]
    return [(x, y) for x in range(D) for y in range(D) if grid[x, y] == 1]

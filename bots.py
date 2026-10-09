import random
from collections import deque
import numpy as np
from .fire import step_fire


def shortest_path(start, goal, grid, forbidden=None):
    if forbidden is None:
        forbidden = set()
    if start == goal:
        return [start]
    D = grid.shape[0]
    q = deque()
    q.append(start)
    prev = {start: None}
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < D and 0 <= ny < D and grid[nx, ny] == 1 and (nx, ny) not in prev and (nx, ny) not in forbidden:
                prev[(nx, ny)] = (x, y)
                if (nx, ny) == goal:
                    # reconstruct
                    path = [goal]
                    cur = goal
                    while prev[cur] is not None:
                        cur = prev[cur]
                        path.append(cur)
                    return list(reversed(path))
                q.append((nx, ny))
    return None


class BotBase:
    def __init__(self):
        self.name = 'base'

    def decide(self, pos, button, grid, burning, q_param, t):
        # return next position (tuple) or None to stay
        raise NotImplementedError()


class Bot1(BotBase):
    def __init__(self):
        super().__init__()
        self.name = 'Bot1_plan_once'
        self.path = None

    def decide(self, pos, button, grid, burning, q_param, t):
        if self.path is None:
            # avoid initial fire cells (burning)
            forbidden = set((i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1]) if burning[i, j])
            self.path = shortest_path(pos, button, grid, forbidden)
            if self.path is None:
                return None
        # follow planned path
        try:
            idx = self.path.index(pos)
            if idx + 1 < len(self.path):
                return self.path[idx + 1]
            else:
                return pos
        except ValueError:
            return None


class Bot2(BotBase):
    def __init__(self):
        super().__init__()
        self.name = 'Bot2_replan'

    def decide(self, pos, button, grid, burning, q_param, t):
        forbidden = set((i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1]) if burning[i, j])
        path = shortest_path(pos, button, grid, forbidden)
        if not path:
            return None
        if len(path) >= 2:
            return path[1]
        return pos


class Bot3(BotBase):
    def __init__(self):
        super().__init__()
        self.name = 'Bot3_buffer'

    def decide(self, pos, button, grid, burning, q_param, t):
        D = grid.shape[0]
        forbidden = set((i, j) for i in range(D) for j in range(D) if burning[i, j])
        # also block neighbors of burning cells
        neighbors = set()
        for x in range(D):
            for y in range(D):
                if burning[x, y]:
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < D and 0 <= ny < D and grid[nx, ny] == 1:
                            neighbors.add((nx, ny))
        forbidden_with_buffer = forbidden.union(neighbors)
        path = shortest_path(pos, button, grid, forbidden_with_buffer)
        if path:
            return path[1] if len(path) >= 2 else pos
        # fallback to only avoiding burning
        path = shortest_path(pos, button, grid, forbidden)
        if path:
            return path[1] if len(path) >= 2 else pos
        return None


class Bot4(BotBase):
    def __init__(self, mc=30, horizon=30):
        super().__init__()
        self.name = 'Bot4_mc_rollout'
        self.mc = mc
        self.horizon = horizon

    def _rollout_success(self, start_pos, candidate_pos, button, grid, burning, q_param):
        # simulate one rollout: bot goes along shortest path from candidate_pos to button (computed at rollout start avoiding burning)
        D = grid.shape[0]
        sim_burning = burning.copy()
        pos = candidate_pos
        # compute a static path at start of rollout avoiding current burning
        forbidden = set((i, j) for i in range(D) for j in range(D) if sim_burning[i, j])
        path = shortest_path(pos, button, grid, forbidden)
        if path is None:
            return False
        # follow path step by step while fire advances
        path_idx = 0
        max_steps = min(self.horizon, len(path) + 5)
        for step in range(max_steps):
            # check collision
            if sim_burning[pos[0], pos[1]]:
                return False
            if pos == button:
                return True
            # move along path
            if path_idx + 1 < len(path):
                path_idx += 1
                pos = path[path_idx]
            # advance fire
            sim_burning = step_fire(grid, sim_burning, q_param)
        return pos == button and not sim_burning[pos[0], pos[1]]

    def decide(self, pos, button, grid, burning, q_param, t):
        # evaluate staying and moving to each neighbor by Monte Carlo rollouts
        D = grid.shape[0]
        candidates = [pos]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < D and 0 <= ny < D and grid[nx, ny] == 1:
                candidates.append((nx, ny))

        best = None
        best_score = -1.0
        for c in candidates:
            succ = 0
            for _ in range(self.mc):
                if self._rollout_success(pos, c, button, grid, burning.copy(), q_param):
                    succ += 1
            score = succ / max(1, self.mc)
            if score > best_score:
                best_score = score
                best = c
        return best

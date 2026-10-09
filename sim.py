import random
import numpy as np
from .ship import get_open_cells
from .fire import init_burning, step_fire


def run_trial(grid, q, bot, max_steps=1000, seed=None):
    """Run a single trial.

    grid: numpy array open mask
    q: flammability
    bot: instance of BotBase
    returns (success:bool, steps:int)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    opens = get_open_cells(grid)
    if len(opens) < 3:
        return False, 0
    # pick distinct cells for bot, button, fire
    bot_pos = tuple(random.choice(opens))
    button = tuple(random.choice(opens))
    while button == bot_pos:
        button = tuple(random.choice(opens))
    fire_cell = tuple(random.choice(opens))
    while fire_cell == bot_pos or fire_cell == button:
        fire_cell = tuple(random.choice(opens))

    burning = init_burning(grid, fire_cell)

    # time steps
    for t in range(1, max_steps + 1):
        # bot decides
        next_pos = bot.decide(bot_pos, button, grid, burning, q, t)
        if next_pos is None:
            # stay
            next_pos = bot_pos
        bot_pos = next_pos
        # check button
        if bot_pos == button:
            return True, t
        # advance fire
        burning = step_fire(grid, burning, q)
        # check collision
        if burning[bot_pos[0], bot_pos[1]]:
            return False, t

    return False, max_steps

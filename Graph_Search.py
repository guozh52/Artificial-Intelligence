from pathlib import Path
from queue import PriorityQueue
from typing import Set, Tuple, List

import numpy as np
import numpy.typing as npt

from hw1.utils import neighbors, plot_GVD, PathPlanMode, distance


def cell_to_GVD_gradient_ascent(
        grid: npt.ArrayLike, GVD: Set[Tuple[int, int]], cell: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """Find the shortest path from any cell in the enviroment to a cell on the
    GVD using gradient ascent.
    Args:
        grid (numpy): NxN numpy array representing the world, with obstacles,
        walls, and the distance from each cell to the obstacles.
        GVD (set[tuple]): A set of tuples containing the cells on the GVD.
        cell (tuple): The starting/ending cell of this path.
    Returns:
        list<tuple>: list of tuples of the path.
    """

    path = [cell]
    # TODO: Implement this method
    while True:
        if path[-1] not in GVD:
            neighbor_cells = neighbors(grid, path[-1][0], path[-1][1])
            expand_cell = path[-1]
            for cell_ in neighbor_cells:
                if grid[cell_[0]][cell_[1]] > grid[expand_cell[0]][expand_cell[1]]:
                    expand_cell = cell_
            path.append(expand_cell)
            pass
        else:
            break

    return path


def cell_to_GVD_a_star(
        grid: npt.ArrayLike, GVD: Set[Tuple[int, int]], cell: Tuple[int, int],
        goal: Tuple[int, int]
):
    """Find the shortest path from any cell in the enviroment to the GVD using
    A* with L2 distance heurstic.
    Args:
        grid (numpy): NxN numpy array representing the world, with obstacles,
        walls, and the distance from each cell to the obstacles.
        GVD (set<tuple>): A set of tuples containing the cells on the GVD.
        cell (tuple): The starting/ending cell of this path.
    Returns:
        list[tuple], dict, list[int]: list of tuples of the path, and the reached 
        dictionary, and the list of frontier sizes. 
    """

    # define a priority queue
    frontier = PriorityQueue()
    frontier.put((0, cell))
    frontier_size = [0]

    # construct a reached table using python dictionary. The key is the (x, y)
    # tuple of the cell position, the value is dictiionary with the cell's cost,
    # and cell parent.
    reached = {cell: {"cost": 0, "parent": None}}

    cell_0 = []

    while not frontier.empty():
        # TODO: implement this
        frontier_size.append(frontier.qsize())
        cell_ = list(frontier.get())
        expand_cell = set(neighbors(grid, cell_[1][0], cell_[1][1])) - set(tuple(reached.keys()))

        for cell_0 in expand_cell:
            if cell_0 in GVD:
                path = [cell_0, tuple(cell_[1])]
                while path[-1] != cell:
                    path.append(reached[path[-1]]["parent"])

                return list(reversed(path)), reached, frontier_size

            else:
                cost = distance([cell_0], cell_[1]) + reached[tuple(cell_[1])]["cost"]  # g value
                frontier.put((cost + distance([cell_0], [goal]), cell_0))
                reached.update({cell_0: {"cost": cost, "parent": cell_[1]}})

    # TODO: implement this to use the reached table (back pointers) to find
    # the path once you have reached a cell on the GVD.

    return None


def GVD_path(
        grid: npt.ArrayLike,
        GVD: Set[Tuple[int, int]],
        A: Tuple[int, int],
        B: Tuple[int, int],
        mode: PathPlanMode):
    """Find the shortest path between two points on the GVD using
    Breadth-First-Search
    Args:
        grid (numpy): NxN numpy array representing the world, with obstacles,
        walls, and the distance from each cell to the obstacles.
        A (tuple): The starting cell of the path.
        B (tuple): The ending cell of the path.
    Returns:
        list[tuple], dict, list[int]: return the path, pointers, and frontier 
        size array. 
    """

    """======Breadth First Search======"""

    # the set of cells on the GVD
    GVD = set(GVD)

    # the set of visited cells
    closed = set([])

    # the set of cells on the current frontier
    # It is a list of coordinates of cell [(x1,y1), (x2,y2)]
    frontier = [A]  # Add the start node to the frontier

    # back pointers to find the path once reached the goal B. The keys
    # should both be tuples of cell positions (x, y)
    pointers = {}

    # the length of the frontier array, update this variable at each step. 
    frontier_size = [0]

    while len(frontier) > 0:
        # TODO:implement this
        """======Breadth First Search======"""
        if mode == PathPlanMode.BFS:
            frontier_size.append(int(len(frontier)))
            cell = frontier.pop(0)
            # neighbors return the list of adjacent cell coordinate [(x1,y1), (x2,y2)]
            expand_cell = set(neighbors(grid, cell[0], cell[1])) - closed

            #Remove all the cell wich not in GVD
            for cell_ in list(expand_cell):
                if cell_ not in GVD:
                    expand_cell.remove(cell_)

            # Remove all the obstacle cell
            for cell_ in list(expand_cell):
                if grid[cell_[0]][cell[1]] <= 0:
                    expand_cell.remove(cell_)

            if B in expand_cell:
                pointers[B] = cell
                path = [B]  # Add B to the top of path
                while path[-1] != A:
                    next_cell = pointers[path[-1]]
                    path.append(next_cell)
                return list(reversed(path)), pointers, frontier_size

            else:
                frontier += list(expand_cell)
                closed.update(expand_cell)
                pointers.update(dict.fromkeys(list(expand_cell), cell))
        # ======Depth First Search======
        elif mode == PathPlanMode.DFS:
            frontier_size.append(int(len(frontier)))
            cell = frontier.pop(0)
            # neighbors return the list of adjacent cell coordinate [(x1,y1), (x2,y2)]
            expand_cell = set(neighbors(grid, cell[0], cell[1])) - closed

            #Remove all the cell which are not in GVD
            for cell_ in list(expand_cell):
                if cell_ not in GVD:
                    expand_cell.remove(cell_)

            # Remove all the obstacle cell
            for cell_ in list(expand_cell):
                if grid[cell_[0]][cell[1]] <= 0:
                    expand_cell.remove(cell_)

            if B in expand_cell:
                pointers[B] = cell
                path = [B]  # Add B to the top of path
                while path[-1] != A:
                    next_cell = pointers[path[-1]]
                    path.append(next_cell)
                return list(reversed(path)), pointers, frontier_size

            else:
                frontier = list(expand_cell) + frontier
                closed.update(expand_cell)
                pointers.update(dict.fromkeys(list(expand_cell), cell))

    return None, None, None


def compute_path(
        grid,
        GVD: set[tuple],
        start: tuple,
        goal: tuple,
        outmode: PathPlanMode = PathPlanMode.GRAD,
        inmode: PathPlanMode = PathPlanMode.BFS):
    """ Compute the path on the grid from start to goal using the methods
    implemented in this file. 
    Returns:
        list: a list of tuples represent the planned path. 
    """

    if outmode == PathPlanMode.GRAD:
        start_path = cell_to_GVD_gradient_ascent(grid, GVD, start)
        end_path = list(reversed(cell_to_GVD_gradient_ascent(grid, GVD, goal)))
    else:
        start_path = cell_to_GVD_a_star(grid, GVD, start, goal)[0]
        end_path = list(reversed(cell_to_GVD_a_star(grid, GVD, goal, start)[0]))
    mid_path, reached, frontier_size = GVD_path(
        grid, GVD, start_path[-1], end_path[0], inmode)
    return start_path + mid_path[1:-1] + end_path


def test_world(
        world_id,
        start,
        goal,
        outmode: PathPlanMode = PathPlanMode.GRAD,
        inmode: PathPlanMode = PathPlanMode.DFS,
        world_dir="worlds"):
    print(f"Testing world {world_id} with modes {inmode} and {outmode}")
    grid = np.load(f"{world_dir}/world_{world_id}.npy")
    GVD = set([tuple(cell) for cell in np.load(
        f"{world_dir}/world_{world_id}_gvd.npy")])
    path = compute_path(grid, GVD, start, goal, outmode=outmode, inmode=inmode)
    print(f"Path length: {len(path)} steps")
    plot_GVD(grid, world_id, GVD, path)

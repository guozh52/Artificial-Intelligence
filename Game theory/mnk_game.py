from pickle import FALSE

import random

import numpy as np
import numpy.typing as npt

from hw2.utils import utility, successors, Node, Tree, GameStrategy


"""
Alpha Beta Search
"""



def max_value(state: npt.ArrayLike, alpha: float, beta: float, k: int):
    """Find the max value given state and return the utility and the game board
    state after the move. Please see Lecture 6 for pseudocode.

    Args:
        state (np.ndarray): the state of the game board, mxn numpy array.
        alpha (float): the alpha value
        beta (float): the beta value
        k (int): the number of consecutive marks
    Returns:
        tuple[float, np.ndarray]: utility value and the board after the move
    """

    # TODO:
    move = None

    if utility(state, k) == None: # Test whether state is a terminal or not; also return game score if yes
        # Not the Terminal
        v = -1000

        actions = successors(state, 'X')
        for action_ in actions:

            v2, a2 = min_value(action_, alpha, beta, k)

            if v2 > v:
                v, move = v2, action_
                alpha = max(alpha, v)
                
            if v >= beta:
                return v, move

        return v, move
    else:
        # Reach the Terminal, 
        return utility(state, k), state

    return None, None


def min_value(state: npt.ArrayLike, alpha: float, beta: float, k: int):
    """Find the min value given state and return the utility and the game board
    state after the move. Please see Lecture 6 for pseudocode.

    Args:
        state (np.ndarray): the state of the game board, mxn numpy array.
        alpha (float): the alpha value
        beta (float): the beta value
        k (int): the number of consecutive marks
    Returns:
        tuple[float, np.ndarray]: utility value and the board after the move
    """

    # TODO:
    move = None

    if utility(state, k) == None:
        # Not the Terminal 
        v = 1000
        

        actions = successors(state, 'O')
        for action_ in actions:

            v2, a2 = max_value(action_, alpha, beta, k)

            if v2 < v:
                v, move = v2, action_
                beta = min(alpha, v)
                
            if v <= alpha:
                return v, move

        return v, move
    else:
        # Reach the Terminal, 
        return utility(state, k), state

    return None, None


"""
Monte Carlo Tree Search
"""


def select(tree: "Tree", state: npt.ArrayLike, k: int, alpha: float):
    """Starting from state, find a terminal node or node with unexpanded
    children. If all children of a node are in tree, move to the one with the
    highest UCT value.

    Args:
        tree (utils.Tree): the search tree
        state (np.ndarray): the game board state
        k (int): the number of consecutive marks
        alpha (float): exploration parameter
    Returns:
        np.ndarray: the game board state
    """

    # TODO:

    while True:
        if utility(state, k) != None:
            # Reached a terminal

            return state

        # Get cureent state(parent) node
        current_node = tree.get(state) 
        # Get all the children of current node
        children = successors(current_node.state, current_node.player) 
    
        for each_ in children:
            if tree.get(each_) == None:
                # Found a unexpanded child
                return state

        # If all the children is in the tree,
        # then calculate each child's UCT value, move to the one with the highest UCT value.
        best_score = 0.0
        best_child = state
    
        for each_ in children:
            # If child's already in the tree but with a parent node different from the current node, ignore it
            parent_state = tree.get(each_).parent
            if parent_state.tobytes() != current_node.state.tobytes():
                continue

            child = tree.get(each_)
            exploit_value = float(child.w) / float(child.N)
            explore_value = alpha * np.sqrt(np.log(current_node.N) / float(child.N))

            UCT_value = exploit_value + explore_value
        
            if UCT_value > best_score:
                best_child = each_
                best_score = UCT_value

        if state.tobytes() == best_child.tobytes():
            print(state)
            return state
            
        state = best_child
    

def expand(tree: "Tree", state: npt.ArrayLike, k: int):
    """Add a child node of state into the tree if it's not terminal and return
    tree and new state, or return current tree and state if it is terminal.

    Args:
        tree (utils.Tree): the search tree
        state (np.ndarray): the game board state
        k (int): the number of consecutive marks
    Returns:
        tuple[utils.Tree, np.ndarray]: the tree and the game state
    """

    # TODO:

    if utility(state, k) != None:
        # Reached a terminal

        return tree, state

    else:
        # It's not a terminal, add a child node of state into the tree
        # Get all the children of current node
        children = successors(state, tree.get(state).player) 
        for each_ in children:
            if tree.get(each_) == None:
                if tree.get(state).player == 'X':
                    next_player = 'O'
                else:
                    next_player = 'X'
                new_state = each_
                tree.add(Node(new_state, state, next_player, 0, 1))
                return tree, new_state

        return tree, state
        

def simulate(state: npt.ArrayLike, player: str, k: int):
    """Run one game rollout from state to a terminal state using random
    playout policy and return the numerical utility of the result.

    Args:
        state (np.ndarray): the game board state
        player (string): the player, `O` or `X`
        k (int): the number of consecutive marks
    Returns:
        float: the utility
    """

    # TODO:
    while utility(state, k) is None:

        state = random.choice(successors(state, player))

        player = "O" if player == "X" else "X"

    return utility(state, k)


def backprop(tree: "Tree", state: npt.ArrayLike, result: float):
    """Backpropagate result from state up to the root.
    All nodes on path have N, number of plays, incremented by 1.
    If result is a win for a node's parent player, w is incremented by 1.
    If result is a draw, w is incremented by 0.5 for all nodes.

    Args:
        tree (utils.Tree): the search tree
        state (np.ndarray): the game board state
        result (float): the result / utility value

    Returns:
        utils.Tree: the game tree
    """

    # TODO:
    
    if result == 1:
        # 'O' win
        while tree.get(state).parent is not None:
            tree.get(state).N += 1
            if tree.get(state).player == 'O':
                tree.get(state).w += 1
            state = tree.get(state).parent

    elif result == -1:
        # 'X' win
        while tree.get(state).parent is not None:
            tree.get(state).N += 1
            if tree.get(state).player == 'X':
                tree.get(state).w += 1
            state = tree.get(state).parent

    else:
        # Draw
        while tree.get(state).parent is not None:
            tree.get(state).N += 1
            tree.get(state).w += 0.5
            state = tree.get(state).parent

    return tree


# ******************************************************************************
# ****************************** ASSIGNMENT ENDS *******************************
# ******************************************************************************


def MCTS(state: npt.ArrayLike, player: str, k: int, rollouts: int, alpha: float):
    # MCTS main loop: Execute MCTS steps rollouts number of times
    # Then return successor with highest number of rollouts
    tree = Tree(Node(state, None, player, 0, 1))

    for i in range(rollouts):
        leaf = select(tree, state, k, alpha)
        tree, new = expand(tree, leaf, k)
        result = simulate(new, tree.get(new).player, k)
        tree = backprop(tree, new, result)

    nxt = None
    plays = 0

    for s in successors(state, tree.get(state).player):
        if tree.get(s).N > plays:
            plays = tree.get(s).N
            nxt = s
    
    return nxt


def ABS(state: npt.ArrayLike, player: str, k: int):
    # ABS main loop: Execute alpha-beta search
    # X is maximizing player, O is minimizing player
    # Then return best move for the given player
    if player == "X":
        value, move = max_value(state, -float("inf"), float("inf"), k)
    else:
        value, move = min_value(state, -float("inf"), float("inf"), k)

    return value, move


def game_loop(
    state: npt.ArrayLike,
    player: str,
    k: int,
    Xstrat: GameStrategy = GameStrategy.RANDOM,
    Ostrat: GameStrategy = GameStrategy.RANDOM,
    rollouts: int = 0,
    mcts_alpha: float = 0.01,
    print_result: bool = False,
):
    # Plays the game from state to terminal
    # If random_opponent, opponent of player plays randomly, else same strategy as player
    # rollouts and alpha for MCTS; if rollouts is 0, ABS is invoked instead
    current = player
    while utility(state, k) is None:
        if current == "X":
            strategy = Xstrat
        else:
            strategy = Ostrat

        if strategy == GameStrategy.RANDOM:
            state = random.choice(successors(state, current))
        elif strategy == GameStrategy.ABS:
            _, state = ABS(state, current, k)
        else:
            state = MCTS(state, current, k, rollouts, mcts_alpha)

        current = "O" if current == "X" else "X"

        if print_result:
            print(state)

    return utility(state, k)

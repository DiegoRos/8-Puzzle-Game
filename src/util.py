"""
Author: Diego Rosenberg (dr3432)

util.py

- Utility functions for reading input files, expanding nodes, reconstructing paths, and saving results for the 8-puzzle solver.

"""


from typing import List
from EightPuzzle import EightPuzzle
from Node import Node

def read_input_file(file_path: str) -> tuple[list[list[int]], list[list[int]]]:
    """Reads the initial and goal states from the input file."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    initial_state = []
    goal_state = []
    reading_initial = True

    for line in lines:
        line = line.strip()
        if line == "":
            reading_initial = False
            continue
        row = list(map(int, line.split()))
        if reading_initial:
            initial_state.append(row)
        else:
            goal_state.append(row)
    
    return initial_state, goal_state


def expand(problem: EightPuzzle, node: Node) -> List[Node]:
    """Expands a node to generate its children."""
    children = []
    for action in problem.actions(node.state):
        new_state = problem.result(node.state, action)
        path_cost = node.path_cost + 1 # Uniform cost for each move

        # Total cost is g(n) + h(n)
        total_cost = path_cost + problem.action_cost(node.state, action, new_state)
        child_node = Node(state=new_state, parent=node, action=action, path_cost=path_cost, total_cost=total_cost)
        children.append(child_node)
    return children

def reconstruct_path(node: Node) -> tuple[List[Node], List[str]]:
    """Reconstructs the path from the root to the given node."""
    path = []
    actions = []
    while node:
        path.append(node)
        actions.append(node.action)
        node = node.parent
    path.reverse()
    actions.reverse()
    return path, actions[1:]  # Exclude the action of the root node which is None


def save_results(game: EightPuzzle, path: List[Node], actions: List[str], expanded_nodes: int, output_path: str):
    """Saves the results to the specified output file."""
    with open(output_path, 'w') as file:
        # Writie Initial State
        for row in game.initial_state:  
            file.write(" ".join(map(str, row)) + "\n")

        file.write("\n")
        # Write Goal State
        for row in game._goal_state:  
            file.write(" ".join(map(str, row)) + "\n")

        file.write("\n")
        # Write depth
        file.write(len(actions).__str__() + "\n")
        # Write number of expanded nodes
        file.write(expanded_nodes.__str__() + "\n")
        # Write actions
        file.write(" ".join(actions))
        file.write("\n")
        # Write path total costs
        for node in path:
            file.write(f"{node.total_cost} ")
        file.write("\n")
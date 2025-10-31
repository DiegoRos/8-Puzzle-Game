"""
Author: Diego Rosenberg (dr3432)

solve.py
- Serves as main entry point for solving the puzzle based on input files and saving results.
- Implementation of the 8-puzzle solver using A* search algorithm with different heuristics.

Instructions to run:
- Ensure you have Python 3.x installed.
- Place this script in the same directory as EightPuzzle.py and Node.py.
- Run the script from the command line with the required arguments:
    python solve.py <root_folder> -i <input_file> [-he <heuristic>] [-o <output_file>] [-v]

"""

import os
import logging
import argparse
from typing import List
from EightPuzzle import EightPuzzle, HeuristicType
from Node import Node
import heapq
import sys
from util import *


def solve(input_path: str, heuristic: HeuristicType = HeuristicType.LINEAR_CONFLICT) -> tuple[List[Node], List[str], int]:
    """Solves the 8-puzzle problem defined in the input file."""
    initial_state, goal_state = read_input_file(input_path)
    logging.debug(f"Initial State: {initial_state}\n")
    logging.debug(f"Goal State: {goal_state}\n")
    game = EightPuzzle(initial_state, goal_state, heuristic=heuristic)
    
    # Count number of exanded nodes
    expanded_nodes = 0

    # Priority queue for the frontier
    frontier: list[tuple[float, Node]] = []

    # Initialize the root node
    root_node = Node(state=game.initial_state, path_cost=0, total_cost=game.heuristic(game.initial_state))
    # Push the root node onto the frontier with its evaluation function value
    heapq.heappush(frontier, (root_node.total_cost, root_node))

    # Save visited states to avoid cycles
    reached: dict[tuple[int, ...]: Node] = {}

    # Add the initial state to reached
    reached[game.seen_state(root_node.state)] = root_node
    expanded_nodes += 1

    while frontier:
        # Pop the node with the lowest total cost
        _, current_node = heapq.heappop(frontier)

        logging.debug(f"Expanding Node: {current_node}")

        # Check if we have reached the goal
        if game.is_goal(current_node.state):
            logging.info("Goal reached!")
            # Reconstruct the path to the goal
            path, actions = reconstruct_path(current_node)
            logging.debug("Solution Path:")
            for step in path:
                logging.debug(step)
            return path, actions, expanded_nodes

        # Expand the current node
        for child in expand(game, current_node):
            seen_state = game.seen_state(child.state)

            # Check if state has not been reached or found a cheaper path
            if seen_state not in reached or child.path_cost < reached[seen_state].path_cost:
                expanded_nodes += 1
                reached[seen_state] = child
                heapq.heappush(frontier, (child.total_cost, child))
                logging.debug(f"Added to frontier: {child}")

    logging.warning("No solution found.")
    return [], [], expanded_nodes

def main():
    parser = argparse.ArgumentParser(description="8-Puzzle Solver")
    parser.add_argument("root_folder", help="Root folder containing data.")
    parser.add_argument("-i", "--input", required=True,
                       help="Target input file to process (e.g., 'input1.txt').")
    parser.add_argument("-he", "--heuristic", type=str, choices=['manhattan', 'linear_conflict'], default='linear_conflict',
                       help="Heuristic to use: 'manhattan' or 'linear_conflict'. Default is 'linear_conflict'.")
    parser.add_argument("-o", "--output", required=False, help="Ouptut file name, without including used heuristic (e.g. output).")
    parser.add_argument("-v", "--verbose", action="store_true", default=False,
                       help="Enable verbose output.")
    
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(message)s')

    root_folder = args.root_folder
    input_file = args.input
    input_file_name, input_file_ext = os.path.splitext(input_file)
    input_path = os.path.join(root_folder, input_file)
    heuristic = HeuristicType.MANHATTAN if args.heuristic == 'manhattan' else HeuristicType.LINEAR_CONFLICT

    if not os.path.exists(input_path):
        logging.error(f"Input file '{input_path}' does not exist.")
        sys.exit(1)

    logging.info(f"Processing input file: {input_path}")
    
    path, actions, expanded_nodes = solve(input_path, heuristic=heuristic)
    used_heuristic_repr = "h1" if heuristic == HeuristicType.MANHATTAN else "h2"
    
    if args.output:
        output_file = args.output + f"{used_heuristic_repr}.txt"
        output_path = os.path.join(root_folder, output_file)
    else:
        output_file = f"output_{input_file_name}{used_heuristic_repr}.txt"
        output_path = os.path.join(root_folder, output_file)

    save_results(EightPuzzle(*read_input_file(input_path), heuristic=heuristic), path, actions, expanded_nodes, output_path)
    logging.info(f"Results saved to: {output_path}")



if __name__ == "__main__":
    main()
    pass
"""
Author: Diego Rosenberg (dr3432)

Node.py
- Defines the Node class used in search algorithms for the 8-puzzle problem.
- Each node represents a state in the search tree along with metadata such as path cost and parent node.
"""

from typing import TypeVar, Generic, Optional
import logging

# Define generic types for state and action
S = TypeVar('S')
A = TypeVar('A')

class Node(Generic[S, A]):
    """
    A class to represent a node in the search tree.
    It contains the state, a reference to its parent node, the action that
    led to this state, and the total path cost from the root to this node.
    """
    def __init__(self,
                 state: S,
                 parent: Optional['Node'] = None,
                 action: Optional[A] = None,
                 path_cost: float = 0,
                 total_cost: float = 0):
        """
        Initializes a Node instance.

        Args:
            state: The state represented by this node.
            parent: The parent node in the search tree.
            action: The action taken to get from the parent to this node.
            path_cost: The total cost from the initial state to this node.
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.total_cost = total_cost

    def __repr__(self) -> str:
        """Provides a developer-friendly string representation of the node."""
        return f"<Node State: {self.state}, Path Cost: {self.path_cost}, Total Cost: {self.total_cost}>"

    def __lt__(self, other: 'Node') -> bool:
        """
        Comparison method for the priority queue.
        Note: The actual priority is determined by the evaluation function 'f'
        in the Best-First Search algorithm, not by the node's path_cost alone.
        This is a placeholder for custom priority queue implementations.
        """
        return self.path_cost < other.path_cost



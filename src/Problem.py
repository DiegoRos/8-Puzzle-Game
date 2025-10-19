from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar


# Define a generic type for the state, which can be any type
S = TypeVar('S')
# Define a generic type for the action
A = TypeVar('A')

class Problem(ABC, Generic[S, A]):
    """
    An abstract class for a formal search problem.
    This class defines the interface that any search problem must implement
    to be solved by the search algorithms.
    """

    @property
    @abstractmethod
    def initial_state(self) -> S:
        """
        Returns the initial state of the problem.
        Corresponds to `problem.INITIAL` in the pseudocode.
        """
        pass

    def action_cost(self, state: S, action: A, next_state: S) -> float:
        """
        A heuristic function that estimates the cost from the current state to the nearest goal.
        This is optional and can be overridden by specific problem implementations.

        Args:
            state: The current state.
        Returns:
            A non-negative estimate of the cost to reach the goal from the current state.
        """

        # Default heuristic returns 1.0, indicating Unifomrm Cost.
        return 1.0
    
    @abstractmethod
    def is_goal(self, state: S) -> bool:
        """
        Checks if a given state is a goal state.
        Corresponds to `problem.IS-GOAL(node.STATE)` in the pseudocode.

        Args:
            state: The state to check.

        Returns:
            True if the state is a goal state, False otherwise.
        """
        pass

    @abstractmethod
    def actions(self, state: S) -> List[A]:
        """
        Returns the set of legal actions for a given state.
        Corresponds to `problem.ACTIONS(s)` in the pseudocode.

        Args:
            state: The state from which to get actions.

        Returns:
            A list of applicable actions.
        """
        pass

    @abstractmethod
    def result(self, state: S, action: A) -> S:
        """
        Returns the state that results from executing an action in a given state.
        This is the transition model.
        Corresponds to `problem.RESULT(s, action)` in the pseudocode.

        Args:
            state: The current state.
            action: The action to perform.

        Returns:
            The new state after the action.
        """
        pass

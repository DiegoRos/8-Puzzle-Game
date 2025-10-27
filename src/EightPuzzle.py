from Problem import Problem
from typing import Optional, List, Callable, TypeAlias
import random
import logging
from enum import Enum

StateType: TypeAlias = List[List[int]]
ActionType: TypeAlias = str

class HeuristicType(Enum):
    MANHATTAN = 1
    LINEAR_CONFLICT = 2

class EightPuzzle(Problem[StateType, ActionType]):
    """
    Representation of the 8-puzzle problem.
    State is represented as a n_rows x n_cols 2d list of integers, where 0 represents the blank tile.
    Actions are represented as strings: 'U', 'D', 'L', 'R'.

    """

    # Possible moves mapped to coordinate changes
    _moves = {'U': (0, -1), 'D': (0, 1), 'L': (-1, 0), 'R': (1, 0)}
    _nrows = 3 # Default puzzle of 3x3
    _ncols = 3 # Default puzzle of 3x3

    # Create type hint for heuristic functions
    HeuristicType = Callable[[StateType], float]

    def __init__(self,
                 initial: Optional[StateType] = None, 
                 goal: Optional[StateType] = None, 
                 heuristic: Optional[HeuristicType] = None):
        """
        Initializes the 8-puzzle with an initial state and an optional goal state.

        Args:
            initial: A n_rows x n_cols 2d list of integers representing the initial state.
            goal: A n_rows x n_cols 2d list of integers representing the goal state. Defaults to the standard goal state.
        """
        custom_game = initial is not None and goal is not None

        if goal is None:
            logging.info("No goal state provided, generating a random goal state.")
            # Generate random goal state if none provided
            self._goal_state = self._generate_random_state()
        else:
            self._goal_state = goal

        # Create dictionary of goal positions for quick lookup
        # Save as {(tile_x, tile_y): index}
        self._goal_state_positions = self.generate_position_map(self._goal_state)

        if initial is None:
            logging.info("No initial state provided, generating a random solvable state.")
            # Create random initial state if none provided
            self._initial_state = self._generate_random_initial_state()

        else:
            self._initial_state = initial

        if custom_game:
            self._nrows = len(initial)
            self._ncols = len(initial[0]) if self._nrows > 0 else 0

        if not self.is_valid_puzzle(self._initial_state, self._goal_state):
            raise ValueError("The provided initial or goal state is not a valid 8-puzzle configuration.")

        if not self._is_solvable(self._initial_state):
            raise ValueError("The provided initial state is not solvable.")
        
        # Set heuristic function using enum.
        if heuristic is None:
            # Base default heuristic to Linear Conflict since it is optimal
            self.heuristic = self.linear_conflict
            self.selected_heuristic = HeuristicType.LINEAR_CONFLICT

        elif heuristic == HeuristicType.MANHATTAN:
            self.heuristic = self.manhattan_distance
            self.selected_heuristic = HeuristicType.MANHATTAN

        elif heuristic == HeuristicType.LINEAR_CONFLICT:
            self.heuristic = self.linear_conflict
            self.selected_heuristic = HeuristicType.LINEAR_CONFLICT

        else:
            raise ValueError(f"Unknown heuristic type: {heuristic}")
        

    def __repr__(self):
        """Provides a developer-friendly string representation of the puzzle."""
        lines = []
        inter_board_space = "     "  # 5 spaces between boards

        # Calculate the required width for each number cell based on the largest number
        max_num = self._nrows * self._ncols - 1
        cell_width = len(str(max_num))

        # Helper function to format a single row to a consistent width
        def format_row(row_data):
            # Right-align each number within its calculated cell width
            formatted_tiles = [
                f"{(str(tile) if tile != 0 else '_'):>{cell_width}}" 
                for tile in row_data
            ]
            return "[" + " ".join(formatted_tiles) + "]"

        # Calculate the total width of one formatted board for header alignment
        sample_row = format_row(self.initial_state[0])
        board_width = len(sample_row) if self._ncols >= 4 else 16

        # Create the header with dynamic padding
        initial_header = "Initial State:"
        goal_header = "Goal State:"
        header = f"{initial_header:<{board_width}}{inter_board_space}{goal_header}"
        lines.append(header)
        lines.append("-" * len(header)) # Add a separator for clarity

        # Format and combine each row from both states
        for i in range(self._nrows):
            initial_row_str = format_row(self.initial_state[i])
            goal_row_str = format_row(self._goal_state[i])
            
            # Left-align the initial row string to ensure the goal row starts consistently
            combined_row = f"{initial_row_str:<{board_width}}{inter_board_space}{goal_row_str}"
            lines.append(combined_row)
            
        return "\n".join(lines) + '\n'

    @property
    def initial_state(self) -> StateType:
        """
        Returns the initial state of the puzzle.

        Returns:
            A initial state.
        """
        return self._initial_state

        
    def is_goal(self, state: StateType) -> bool:
        """
        Checks if a given state is the goal state.

        Args:
            state: A n_rows x n_cols 2d list of integers representing the state to check.
        Returns:
            True if the state is the goal state, False otherwise.
        """
        return state == self._goal_state
        
    def is_valid_state(self, state: StateType) -> bool:
        """
        Checks if a given state is valid (contains all tiles exactly once).

        Args:
            state: A n_rows x n_cols 2d list of integers representing the state to check.
        Returns:
            True if the state is valid, False otherwise.
        """
        flat_state = [num for row in state for num in row]
        expected_tiles = set(range(self._nrows * self._ncols))
        return set(flat_state) == expected_tiles

    def is_valid_puzzle(self, initial_state: StateType, goal: StateType) -> bool:
        """Checks if the given state is a valid 8-puzzle configuration."""
        # Check equal rows and columns of initial and goal states
        bool_valid_rows = len(initial_state) == len(goal)
        bool_valid_cols = all(len(row) == len(goal[idx]) for idx, row in enumerate(initial_state))
        
        logging.debug(f"Valid Rows: {bool_valid_rows}, Valid Cols: {bool_valid_cols}")

        # Check correct set of numbers (if n x m puzzle then numbers 0 to n*m-1)
        n = len(initial_state)
        m = len(initial_state[0]) if n > 0 else 0
        expected_numbers = set(range(n * m))
        initial_numbers = set(num for row in initial_state for num in row)
        goal_numbers = set(num for row in goal for num in row)

        bool_valid_numbers_initial = initial_numbers == expected_numbers
        bool_valid_numbers_goal = goal_numbers == expected_numbers

        logging.debug(f"Valid Numbers Initial: {bool_valid_numbers_initial}, Valid Numbers Goal: {bool_valid_numbers_goal}")

        return bool_valid_rows and bool_valid_cols and bool_valid_numbers_initial and bool_valid_numbers_goal

    def actions(self, state: StateType) -> List[ActionType]:
        """
        Returns the list of legal actions for a given state.

        Args:
            state: A n_rows x n_cols 2d list of integers representing the current state.
        Returns:
            A list of strings representing the legal actions ('U', 'D', 'L', 'R').
        """
        actions = []
        zero_row, zero_col = self._find_zero_position(state)
        logging.debug(f"Blank tile at position: ({zero_col}, {zero_row})")
        if zero_row > 0:
            actions.append('U')
        if zero_row < self._nrows - 1:
            actions.append('D')
        if zero_col > 0:
            actions.append('L')
        if zero_col < self._ncols - 1:
            actions.append('R')

        return actions

    def result(self, state: StateType, action: ActionType) -> StateType:
        """
        Returns the state that results from executing an action in a given state.

        Args:
            state: A n_rows x n_cols 2d list of integers representing the current state.
            action: A string representing the action to perform ('U', 'D', 'L', 'R').
        Returns:
            A n_rows x n_cols 2d list of integers representing the new state after the action.
        """
        row, col = self._find_zero_position(state)
        move = self._moves[action]
        new_row, new_col = row + move[1], col + move[0]

        # Check if new position is within bounds
        #   Actions passed to this method should already be validated but good to check again (in case of future use).
        if 0 <= new_row < self._nrows and 0 <= new_col < self._ncols:
            new_state = [list(r) for r in state]  # Create a copy
            # Swap the blank tile with the target tile
            new_state[row][col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[row][col]

            return new_state
        else:
            raise ValueError(f"Action '{action}' is not valid from state {state}.")

    def action_cost(self, state: StateType, action: ActionType, next_state: StateType) -> float:
        """
        Returns the cost of applying an action to get from one state to the next.
        In the 8-puzzle, each move has a uniform cost of 1.

        Args:
            state: A n_rows x n_cols 2d list of integers representing the current state.
            action: A string representing the action taken ('U', 'D', 'L', 'R').
            next_state: A n_rows x n_cols 2d list of integers representing the resulting state.
        Returns:
            The cost of the action, which is always 1.0 + the selected heuristic.
        """
        return self.heuristic(next_state)

    def generate_position_map(self, state: StateType) -> dict[int, tuple[int, int]]:
        """
        Generates a mapping from tile values to their positions in the given state.

        Args:
            state: A n_rows x n_cols 2d list of integers representing the state.
        Returns:
            A dictionary mapping tile values to their (col, row) positions.
        """
        position_map = {}
        for row_index, row in enumerate(state):
            for col_index, value in enumerate(row):
                if value == 0:
                    continue  # Skip the blank tile
                position_map[value] = (col_index, row_index)

        return position_map
    
    def seen_state(self, state: StateType) -> tuple[int, ...]:
        """
        Converts a 2D state into a hashable tuple representation.

        Args:
            state: A n_rows x n_cols 2d list of integers representing the state.
        Returns:
            A tuple of integers representing the flattened state.
        """
        return tuple(num for row in state for num in row)

    def _find_zero_position(self, state: StateType) -> tuple[int, int]:
        """
        Finds the position of the blank tile (0) in the given state.

        Args:
            state: A n_rows x n_cols 2d list of integers representing the state.
        Returns:
            A tuple (col, row) representing the position of the blank tile.
        """
        for row_index, row in enumerate(state):
            for col_index, value in enumerate(row):
                if value == 0:
                    return (row_index, col_index)
        raise ValueError("The state does not contain a blank tile (0).")
    

    def _is_solvable(self, state: StateType) -> bool:
        """
        Determines if a given state is solvable.

        Args:
            state: A n_rows x n_cols 2d list of integers representing the state.
        Returns:
            True if the state is solvable, False otherwise.
        """
        if self._goal_state is None:
            return True  # Assume solvable if no goal state is defined
        
        # Define a lambda function to flatten the array
        flatten_array = lambda arr: [item for sublist in arr for item in sublist if item != 0]

        initial_inversions = 0
        flat_initial = flatten_array(state)
        goal_inversions = 0
        flat_goal = flatten_array(self._goal_state)

        # Logging debug flattened states
        logging.debug(f"Flattened Initial State (no 0): {flat_initial}")
        logging.debug(f"Flattened Goal State (no 0): {flat_goal}")

        # Count number of inversions in the initial state
        for i in range(len(flat_initial)):
            for j in range(i + 1, len(flat_initial)):
                if flat_initial[i] > flat_initial[j]:
                    initial_inversions += 1
        
        # Count number of inversions in the goal state
        for i in range(len(flat_goal)):
            for j in range(i + 1, len(flat_goal)):
                if flat_goal[i] > flat_goal[j]:
                    goal_inversions += 1

        logging.debug(f"Initial Inversions: {initial_inversions}, Goal Inversions: {goal_inversions}")

        # If parity of both states is equal, then game is solvable.
        return (initial_inversions % 2) == (goal_inversions % 2)

    def _generate_random_state(self) -> tuple[tuple[int, ...], ...]:
        """
        Generates a random state for the 8-puzzle.

        Returns:
            A 2D tuple of dimensions nrows x ncols integers representing a random solvable state.
        """
        state = list(range(self._ncols * self._nrows))
        random.shuffle(state)

        # Convert to 3x3 list
        state = [state[i * 3:(i + 1) * 3] for i in range(self._nrows)]

        # Return as tuple
        return tuple(state)
            
    
    def _generate_random_initial_state(self) -> tuple[tuple[int, ...], ...]:
        """
        Generates a random solvable state for the 8-puzzle.

        Returns:
            A 2D tuple of dimensions nrows x ncols integers representing a random  state.
        """
        import numpy as np

        state = np.arange(self._nrows, self._ncols).reshape((3, 3))

        # Convert to 3x3 tuple
        while True:
            np.random.permutation(state)
            if self._is_solvable(state):
                return tuple(state)
    
    def manhattan_distance(self, state: StateType) -> float:
        """
        Computes the Manhattan distance heuristic for the 8-puzzle.

        Args:
            state: A n_rows x n_cols 2d list of integers representing the current state of the puzzle.
        Returns:
            A float representing the total Manhattan distance of all tiles from their goal positions.
        """
        distance = 0
        
        # Compute positions fro each tile in state for 3x3 grid
        state_positions = self.generate_position_map(state)

        # Calculate Manhattan distance for each tile
        for tile_value, (current_x, current_y) in state_positions.items():
            if tile_value == 0:
                continue  # Skip the blank tile
            goal_x, goal_y = self._goal_state_positions[tile_value]
            distance += abs(current_x - goal_x) + abs(current_y - goal_y)

        return float(distance)
    
    def _linear_conflict_count(self, state: StateType) -> int:
        """
        Helper method to compute the number of linear conflicts in the given state.

        Args:
            state: A n_rows x n_cols 2d list of integers representing the current state of the puzzle.
        Returns:
            An integer representing the number of linear conflicts.
        """
        conflicts = 0

        # Check for row conflicts
        for r in range(self._nrows):
            for c1 in range(self._ncols):
                tile1 = state[r][c1]
                # Skip blank tile or tiles not in their goal row
                if tile1 == 0 or self._goal_state_positions[tile1][0] != r:
                    continue

                for c2 in range(c1 + 1, self._ncols):
                    tile2 = state[r][c2]
                    # Skip blank tile or tiles not in their goal row
                    if tile2 == 0 or self._goal_state_positions[tile2][0] != r:
                        continue
                    
                    # Check if their relative order is inverted from the goal
                    if self._goal_state_positions[tile1][1] > self._goal_state_positions[tile2][1]:
                        conflicts += 1

        # Check for column conflicts
        for c in range(self._nrows):
            for r1 in range(self._ncols):
                tile1 = state[r1][c]
                # Skip blank tile or tiles not in their goal column
                if tile1 == 0 or self._goal_state_positions[tile1][1] != c:
                    continue

                for r2 in range(r1 + 1, self._nrows):
                    tile2 = state[r2][c]
                    # Skip blank tile or tiles not in their goal column
                    if tile2 == 0 or self._goal_state_positions[tile2][1] != c:
                        continue

                    # Check if their relative order is inverted from the goal
                    if self._goal_state_positions[tile1][0] > self._goal_state_positions[tile2][0]:
                        conflicts += 1
        
        return conflicts

    def linear_conflict(self, state: tuple[int, ...]) -> float:
        """
        Computes the Linear Conflict heuristic for the 8-puzzle.

        Args:
            state: A n_rows x n_cols 2d list of integers representing the current state of the puzzle.  
        Returns:
            A float representing the total Linear Conflicts of all tiles from their goal positions.
        """
        return self.manhattan_distance(state) + self._linear_conflict_count(state) * 2


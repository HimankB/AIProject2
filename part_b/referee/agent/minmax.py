# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent (Minimax Version)

from referee.game import PlayerColor, Coord, Direction, Action, MoveAction, GrowAction
from referee.game.board import Board
from referee.game.constants import BOARD_N
import time

class Agent:
    """
    Entry point for Freckers game-playing agent using minimax.
    This version implements minimax without alpha-beta pruning for testing purposes.
    """
    def __init__(self, color: PlayerColor, **referee: dict):
        self._color = color
        self._opponent = color.opponent
        # Internal game board representation
        self.board = Board()
        # Maximum search depth
        self.max_depth = 3
        # Time limit per move (seconds)
        self.time_limit = referee.get("time_remaining", None)

    def action(self, **referee: dict) -> Action:
        # Update time limit and record start time
        self.time_limit = referee.get("time_remaining", None)
        start = time.time()
        best_move = GrowAction()
        # Iterative deepening
        for depth in range(1, self.max_depth + 1):
            try:
                score, move = self.minimax(self.board, depth, True, start)
                if move is not None:
                    best_move = move
            except TimeoutError:
                break
            # Time cutoff: leave some buffer
            if self.time_limit and time.time() - start > self.time_limit * 0.9:
                break
        return best_move

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        # Apply the action to update internal game state
        self.board.apply_action(action)

    def minimax(self, board: Board, depth: int, maximizing: bool, start_time: float):
        # Time check
        if self.time_limit and time.time() - start_time > self.time_limit * 0.9:
            raise TimeoutError

        if depth == 0 or board.game_over:
            return self.evaluate(board), None

        player = self._color if maximizing else self._opponent
        
        # Generate all legal moves including multi-jumps
        moves = self.generate_moves(board, player)
        
        if maximizing:
            best_score = float('-inf')
            best_move = None
            
            for move in moves:
                # Time check before exploring child
                if self.time_limit and time.time() - start_time > self.time_limit * 0.9:
                    break
                    
                # Try to apply the move
                try:
                    board.apply_action(move)
                    score, _ = self.minimax(board, depth - 1, False, start_time)
                    board.undo_action()
                    
                    if score > best_score:
                        best_score = score
                        best_move = move
                except Exception:
                    # If the move is illegal, simply skip it
                    continue
                    
            return best_score, best_move
        else:
            best_score = float('inf')
            best_move = None
            
            for move in moves:
                # Time check before exploring child
                if self.time_limit and time.time() - start_time > self.time_limit * 0.9:
                    break
                    
                # Try to apply the move
                try:
                    board.apply_action(move)
                    score, _ = self.minimax(board, depth - 1, True, start_time)
                    board.undo_action()
                    
                    if score < best_score:
                        best_score = score
                        best_move = move
                except Exception:
                    # If the move is illegal, simply skip it
                    continue
                    
            return best_score, best_move

    def generate_moves(self, board: Board, player: PlayerColor):
        """
        Generate legal moves: 
        - GrowAction
        - All single-step MoveActions 
        - All multi-jump MoveActions
        """
        moves = [GrowAction()]
        
        # Find all frogs of the player
        for coord in board._occupied_coords():
            if board[coord].state != player:
                continue
            
            # Add single step moves
            for direction in Direction:
                try:
                    # Skip illegal directions based on player color
                    if (player == PlayerColor.RED and direction in [Direction.Up, Direction.UpRight, Direction.UpLeft]) or \
                       (player == PlayerColor.BLUE and direction in [Direction.Down, Direction.DownRight, Direction.DownLeft]):
                        continue
                    
                    target = coord + direction
                    # Single step move to an empty lily pad
                    if board._within_bounds(target) and board[target].state == "LilyPad":
                        move = MoveAction(coord, [direction])
                        moves.append(move)
                except Exception:
                    continue
            
            # Find multi-jump moves recursively
            self._find_jumps(board, player, coord, [], [], moves)
                
        return moves
    
    def _find_jumps(self, board: Board, player: PlayerColor, current: Coord, 
                    path: list[Direction], visited: list[Coord], moves: list[Action]):
        """
        Recursively find all possible jump sequences from the current position.
        
        Args:
            board: Current game board
            player: Current player
            current: Current coordinate
            path: List of directions taken so far
            visited: List of coordinates visited so far
            moves: List to store valid moves
        """
        for direction in Direction:
            # Skip illegal directions based on player color
            if (player == PlayerColor.RED and direction in [Direction.Up, Direction.UpRight, Direction.UpLeft]) or \
               (player == PlayerColor.BLUE and direction in [Direction.Down, Direction.DownRight, Direction.DownLeft]):
                continue
                
            try:
                # Check for a jump possibility: need any frog (friend or enemy) followed by an empty lily pad
                middle = current + direction
                if not board._within_bounds(middle) or not board._cell_occupied_by_player(middle):
                    continue
                # Note: _cell_occupied_by_player checks for any player's frog (RED or BLUE), which is what we want
                    
                target = middle + direction
                if not board._within_bounds(target) or target in visited or board[target].state != "LilyPad":
                    continue
                
                # Found a valid jump
                new_path = path + [direction]
                new_visited = visited + [target]
                
                # Add this jump sequence as a move
                moves.append(MoveAction(current, new_path))
                
                # Continue looking for more jumps from the new position
                self._find_jumps(board, player, target, new_path, new_visited, moves)
                
            except Exception:
                continue

    def evaluate(self, board: Board):
        """
        Heuristic combining:
          - Goal-row frog difference (×100)
          - Distance-to-goal differential (×5)
          - Mobility score (x2)
        """
        # Goal-row score
        my_goal = board._player_score(self._color)
        opp_goal = board._player_score(self._opponent)
        goal_diff = my_goal - opp_goal

        # Distance-to-goal differential
        my_dist = 0
        opp_dist = 0
        my_frogs = 0
        opp_frogs = 0
        
        for coord in board._occupied_coords():
            state = board[coord].state
            row = coord.r
            if state == self._color:
                my_frogs += 1
                if self._color == PlayerColor.RED:
                    my_dist += (BOARD_N - 1 - row)
                else:
                    my_dist += row
            elif state == self._opponent:
                opp_frogs += 1
                if self._opponent == PlayerColor.RED:
                    opp_dist += (BOARD_N - 1 - row)
                else:
                    opp_dist += row
        
        dist_diff = opp_dist - my_dist
        
        # Mobility score: reward having more frogs
        mobility_diff = my_frogs - opp_frogs

        # Composite evaluation
        return (100 * goal_diff) + (5 * dist_diff) + (2 * mobility_diff)
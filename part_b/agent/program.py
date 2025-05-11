# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Coord, Direction, Action, MoveAction, GrowAction
from referee.game.board import Board
from referee.game.constants import BOARD_N
import time

class Agent:
    """
    Entry point for Freckers game-playing agent using alpha-beta pruning.
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
            alpha = float('-inf')
            beta = float('inf')
            try:
                score, move = self.alpha_beta(self.board, depth, alpha, beta, True, start)
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

    def alpha_beta(self, board: Board, depth: int, alpha: float, beta: float, maximizing: bool, start_time: float):
        # Time check
        if self.time_limit and time.time() - start_time > self.time_limit * 0.9:
            raise TimeoutError

        if depth == 0 or board.game_over:
            return self.evaluate(board), None

        best_move = None
        player = self._color if maximizing else self._opponent
        for move in self.generate_moves(board, player):
            # Time check before exploring child
            if self.time_limit and time.time() - start_time > self.time_limit * 0.9:
                break
            board.apply_action(move)
            score, _ = self.alpha_beta(board, depth - 1, alpha, beta, not maximizing, start_time)
            board.undo_action()

            if maximizing:
                if score > alpha:
                    alpha = score
                    best_move = move
            else:
                if score < beta:
                    beta = score
                    best_move = move

            # Prune
            if alpha >= beta:
                break

        return (alpha, best_move) if maximizing else (beta, best_move)

    def generate_moves(self, board: Board, player: PlayerColor):
        """
        Generate legal moves: all single-step MoveActions and GrowAction.
        """
        moves = [GrowAction()]
        for coord in board._occupied_coords():
            if board[coord].state != player:
                continue
            for direction in Direction:
                try:
                    move = MoveAction(coord, [direction])
                    board.apply_action(move)
                    moves.append(move)
                    board.undo_action()
                except Exception:
                    continue
        return moves

    def evaluate(self, board: Board):
        """
        Heuristic combining:
          - Goal-row frog difference (×100)
          - Distance-to-goal differential (×5)
        """
        # Goal-row score
        my_goal = board._player_score(self._color)
        opp_goal = board._player_score(self._opponent)
        goal_diff = my_goal - opp_goal

        # Distance-to-goal differential
        my_dist = 0
        opp_dist = 0
        for coord in board._occupied_coords():
            state = board[coord].state
            row = coord.r
            if state == self._color:
                if self._color == PlayerColor.RED:
                    my_dist += (BOARD_N - 1 - row)
                else:
                    my_dist += row
            elif state == self._opponent:
                if self._opponent == PlayerColor.RED:
                    opp_dist += (BOARD_N - 1 - row)
                else:
                    opp_dist += row
        dist_diff = opp_dist - my_dist

        # Composite evaluation
        return (100 * goal_diff) + (5 * dist_diff)

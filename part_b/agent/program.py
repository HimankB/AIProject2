# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Coord, Direction, Action, MoveAction, GrowAction
from referee.game.board import Board
from referee.game.constants import BOARD_N
import time


class Agent:
    """
    Entry point for Freckers game-playing agent using alpha-beta pruning.
    Enhanced to support multi-jump moves.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        self._color = color
        self._opponent = color.opponent
        # Internal game board representation
        self.board = Board()
        # Maximum search depth
        self.max_depth = 4
        # Time limit per move (seconds)
        self.time_limit = referee.get("time_remaining", None)

    def action(self, **referee: dict) -> Action:
        # Update time limit and record start time
        self.time_limit = referee.get("time_remaining", None)
        start = time.time()
        best_move = GrowAction()
        # Iterative deepening
        for depth in range(1, self.max_depth + 1):
            alpha = float("-inf")
            beta = float("inf")
            try:
                score, move = self.alpha_beta(
                    self.board, depth, alpha, beta, True, start
                )
                if move is not None:
                    best_move = move
            except TimeoutError:
                break
            # Time cutoff: leave some buffer
            if self.time_limit and time.time() - start > self.time_limit * 0.9:
                break
        return best_move

    def update(self, color: PlayerColor, action: Action, **referee):
        # sync the turn
        self.board._turn_color = color

        # now apply (and let the board flip to the next turn itself)
        self.board.apply_action(action)

    def alpha_beta(
        self,
        board: Board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        start_time: float,
    ):
        # Time check
        if self.time_limit and time.time() - start_time > self.time_limit * 0.9:
            raise TimeoutError

        if depth == 0 or board.game_over:
            return self.evaluate(board), None

        best_move = None
        player = self._color if maximizing else self._opponent

        # Generate all legal moves including multi-jumps
        moves = self.generate_moves(board, player)

        for move in moves:
            # Time check before exploring child
            if self.time_limit and time.time() - start_time > self.time_limit * 0.9:
                break

            # Try to apply the move
            try:
                board.apply_action(move)
                score, _ = self.alpha_beta(
                    board, depth - 1, alpha, beta, not maximizing, start_time
                )
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
            except Exception:
                # If the move is illegal, simply skip it
                continue

        return (alpha, best_move) if maximizing else (beta, best_move)

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
                    if (
                        player == PlayerColor.RED
                        and direction
                        in [Direction.Up, Direction.UpRight, Direction.UpLeft]
                    ) or (
                        player == PlayerColor.BLUE
                        and direction
                        in [Direction.Down, Direction.DownRight, Direction.DownLeft]
                    ):
                        continue

                    target = coord + direction
                    # Single step move to an empty lily pad
                    if (
                        board._within_bounds(target)
                        and board[target].state == "LilyPad"
                    ):
                        move = MoveAction(coord, [direction])
                        moves.append(move)
                except Exception:
                    continue

            # Find multi-jump moves recursively
            self._find_jumps(board, player, coord, [], [], moves)

        return moves

    def _find_jumps(
        self,
        board: Board,
        player: PlayerColor,
        current: Coord,
        path: list[Direction],
        visited: list[Coord],
        moves: list[Action],
    ):
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
            if (
                player == PlayerColor.RED
                and direction in [Direction.Up, Direction.UpRight, Direction.UpLeft]
            ) or (
                player == PlayerColor.BLUE
                and direction
                in [Direction.Down, Direction.DownRight, Direction.DownLeft]
            ):
                continue

            try:
                # Check for a jump possibility: need any frog (friend or enemy) followed by an empty lily pad
                middle = current + direction
                if not board._within_bounds(
                    middle
                ) or not board._cell_occupied_by_player(middle):
                    continue
                # Note: _cell_occupied_by_player checks for any player's frog (RED or BLUE), which is what we want

                target = middle + direction
                if (
                    not board._within_bounds(target)
                    or target in visited
                    or board[target].state != "LilyPad"
                ):
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
        Enhanced heuristic combining:
          - Goal-row frog difference (×100)
          - Distance-to-goal differential (×5)
          - Mobility score (×2)
          - Board control (normalized, ×1)
          - Jump opportunities (×4)
        """
        # 1) Goal‐row difference
        my_goal = board._player_score(self._color)
        opp_goal = board._player_score(self._opponent)
        goal_diff = my_goal - opp_goal

        # 2) Counters for other features
        my_dist = opp_dist = 0
        my_frogs = opp_frogs = 0
        my_jump_ops = opp_jump_ops = 0
        my_nearby_lilypads = opp_nearby_lilypads = 0

        for coord in board._occupied_coords():
            state = board[coord].state
            row = coord.r

            # Whose frog?
            if state == self._color:
                my_frogs += 1
                my_dist += (
                    (BOARD_N - 1 - row) if self._color == PlayerColor.RED else row
                )

                # look at every direction for control & jumps
                for d in Direction:
                    # skip backwards for each color
                    if (
                        self._color == PlayerColor.RED
                        and d in (Direction.Up, Direction.UpLeft, Direction.UpRight)
                    ) or (
                        self._color == PlayerColor.BLUE
                        and d
                        in (Direction.Down, Direction.DownLeft, Direction.DownRight)
                    ):
                        continue

                    # try to step one
                    try:
                        step = coord + d
                    except ValueError:
                        continue

                    # board control: empty pad next to you
                    if board._within_bounds(step) and board[step].state == "LilyPad":
                        my_nearby_lilypads += 1

                    # jump: frog then empty pad
                    # need to catch out-of-bounds on second hop
                    try:
                        mid = step
                        jump_tgt = mid + d
                    except ValueError:
                        continue

                    if (
                        board._within_bounds(mid)
                        and board._cell_occupied_by_player(mid)
                        and board._within_bounds(jump_tgt)
                        and board[jump_tgt].state == "LilyPad"
                    ):
                        my_jump_ops += 1

            elif state == self._opponent:
                opp_frogs += 1
                opp_dist += (
                    (BOARD_N - 1 - row) if self._opponent == PlayerColor.RED else row
                )

                for d in Direction:
                    if (
                        self._opponent == PlayerColor.RED
                        and d in (Direction.Up, Direction.UpLeft, Direction.UpRight)
                    ) or (
                        self._opponent == PlayerColor.BLUE
                        and d
                        in (Direction.Down, Direction.DownLeft, Direction.DownRight)
                    ):
                        continue

                    try:
                        step = coord + d
                    except ValueError:
                        continue

                    if board._within_bounds(step) and board[step].state == "LilyPad":
                        opp_nearby_lilypads += 1

                    try:
                        mid = step
                        jump_tgt = mid + d
                    except ValueError:
                        continue

                    if (
                        board._within_bounds(mid)
                        and board._cell_occupied_by_player(mid)
                        and board._within_bounds(jump_tgt)
                        and board[jump_tgt].state == "LilyPad"
                    ):
                        opp_jump_ops += 1

        # 3) Feature differentials
        dist_diff = opp_dist - my_dist
        mobility_diff = my_frogs - opp_frogs
        jump_diff = my_jump_ops - opp_jump_ops

        # 4) Normalize board control (pads per frog)
        my_ctrl = (my_nearby_lilypads / my_frogs) if my_frogs else 0
        opp_ctrl = (opp_nearby_lilypads / opp_frogs) if opp_frogs else 0
        control_diff = my_ctrl - opp_ctrl

        # 5) Weighted sum
        W = {
            "goal": 100,
            "dist": 10,
            "mob": 2,
            "ctrl": 1,  # adjust down if still too strong
            "jump": 5,
        }

        return (
            W["goal"] * goal_diff
            + W["dist"] * dist_diff
            + W["mob"] * mobility_diff
            + W["ctrl"] * control_diff
            + W["jump"] * jump_diff
        )

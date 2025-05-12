from referee.game import PlayerColor, GrowAction

class Agent:
    def __init__(self, color: PlayerColor, **referee: dict):
        self._color = color

    def action(self, **referee: dict):
        return GrowAction()  # Always grows, never moves

    def update(self, color, action, **referee: dict):
        pass

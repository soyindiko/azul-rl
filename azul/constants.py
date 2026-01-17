"""
Constants for the Azul board game.
"""

from enum import IntEnum
from typing import List, Tuple

# Number of players (standard game supports 2-4)
MIN_PLAYERS = 2
MAX_PLAYERS = 4

# Tile colors
class TileColor(IntEnum):
    BLUE = 0
    YELLOW = 1
    RED = 2
    BLACK = 3
    WHITE = 4
    # Special marker (first player token)
    FIRST_PLAYER = 5
    # Empty/no tile
    EMPTY = 6

NUM_TILE_COLORS = 5  # Excluding FIRST_PLAYER and EMPTY

# Total tiles per color in the bag
TILES_PER_COLOR = 20

# Factory displays
FACTORIES_BY_PLAYERS = {
    2: 5,
    3: 7,
    4: 9
}
TILES_PER_FACTORY = 4

# Player board dimensions
PATTERN_LINES = 5  # 5 rows (1-5 spaces each)
WALL_SIZE = 5      # 5x5 wall grid

# Floor line (penalty area)
FLOOR_LINE_SIZE = 7
FLOOR_PENALTIES = [-1, -1, -2, -2, -2, -3, -3]

# Wall pattern (standard Azul wall)
# Each row has a specific color pattern
WALL_PATTERN: List[List[TileColor]] = [
    [TileColor.BLUE, TileColor.YELLOW, TileColor.RED, TileColor.BLACK, TileColor.WHITE],
    [TileColor.WHITE, TileColor.BLUE, TileColor.YELLOW, TileColor.RED, TileColor.BLACK],
    [TileColor.BLACK, TileColor.WHITE, TileColor.BLUE, TileColor.YELLOW, TileColor.RED],
    [TileColor.RED, TileColor.BLACK, TileColor.WHITE, TileColor.BLUE, TileColor.YELLOW],
    [TileColor.YELLOW, TileColor.RED, TileColor.BLACK, TileColor.WHITE, TileColor.BLUE],
]

# Bonus points at end of game
BONUS_COMPLETE_ROW = 2
BONUS_COMPLETE_COLUMN = 7
BONUS_COMPLETE_COLOR = 10

# Game end condition
GAME_END_TRIGGER = 1  # Game ends when any player completes at least 1 horizontal row


def get_wall_position(row: int, color: TileColor) -> Tuple[int, int]:
    """Get the column position for a color in a specific wall row."""
    for col, tile_color in enumerate(WALL_PATTERN[row]):
        if tile_color == color:
            return (row, col)
    raise ValueError(f"Color {color} not found in row {row}")


def get_color_at_wall_position(row: int, col: int) -> TileColor:
    """Get the color at a specific wall position."""
    return WALL_PATTERN[row][col]

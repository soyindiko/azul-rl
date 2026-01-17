"""
Core game logic for Azul board game.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from copy import deepcopy
import random

from azul.constants import (
    TileColor, NUM_TILE_COLORS, TILES_PER_COLOR, TILES_PER_FACTORY,
    FACTORIES_BY_PLAYERS, PATTERN_LINES, WALL_SIZE, FLOOR_LINE_SIZE,
    FLOOR_PENALTIES, WALL_PATTERN, BONUS_COMPLETE_ROW, BONUS_COMPLETE_COLUMN,
    BONUS_COMPLETE_COLOR, get_wall_position
)


@dataclass
class PlayerBoard:
    """Represents a single player's board state."""
    
    # Pattern lines: list of (current_count, color) for each row
    # Row i can hold i+1 tiles
    pattern_lines: List[Tuple[int, Optional[TileColor]]] = field(default_factory=list)
    
    # Wall: 5x5 grid, True if tile placed
    wall: np.ndarray = field(default_factory=lambda: np.zeros((WALL_SIZE, WALL_SIZE), dtype=bool))
    
    # Floor line: list of tile colors (including FIRST_PLAYER marker)
    floor_line: List[TileColor] = field(default_factory=list)
    
    # Current score
    score: int = 0
    
    def __post_init__(self):
        if not self.pattern_lines:
            # Initialize empty pattern lines
            self.pattern_lines = [(0, None) for _ in range(PATTERN_LINES)]
    
    def can_place_on_pattern_line(self, row: int, color: TileColor) -> bool:
        """Check if tiles of given color can be placed on pattern line row."""
        if row < 0 or row >= PATTERN_LINES:
            return False
        
        current_count, current_color = self.pattern_lines[row]
        max_capacity = row + 1
        
        # Check if row is full
        if current_count >= max_capacity:
            return False
        
        # Check if color matches or row is empty
        if current_color is not None and current_color != color:
            return False
        
        # Check if this color is already on the wall in this row
        wall_col = None
        for col, wall_color in enumerate(WALL_PATTERN[row]):
            if wall_color == color:
                wall_col = col
                break
        
        if wall_col is not None and self.wall[row, wall_col]:
            return False
        
        return True
    
    def place_on_pattern_line(self, row: int, color: TileColor, count: int) -> int:
        """
        Place tiles on pattern line. Returns number of tiles that overflow to floor.
        """
        current_count, _ = self.pattern_lines[row]
        max_capacity = row + 1
        
        tiles_to_place = min(count, max_capacity - current_count)
        overflow = count - tiles_to_place
        
        self.pattern_lines[row] = (current_count + tiles_to_place, color)
        
        return overflow
    
    def add_to_floor(self, tiles: List[TileColor]) -> None:
        """Add tiles to floor line."""
        for tile in tiles:
            if len(self.floor_line) < FLOOR_LINE_SIZE:
                self.floor_line.append(tile)
            # Excess tiles are discarded (go to box lid in real game)
    
    def score_tile_placement(self, row: int, col: int) -> int:
        """Calculate score for placing a tile at wall position."""
        points = 0
        
        # Count horizontal adjacent tiles
        h_count = 1
        # Left
        for c in range(col - 1, -1, -1):
            if self.wall[row, c]:
                h_count += 1
            else:
                break
        # Right
        for c in range(col + 1, WALL_SIZE):
            if self.wall[row, c]:
                h_count += 1
            else:
                break
        
        # Count vertical adjacent tiles
        v_count = 1
        # Up
        for r in range(row - 1, -1, -1):
            if self.wall[r, col]:
                v_count += 1
            else:
                break
        # Down
        for r in range(row + 1, WALL_SIZE):
            if self.wall[r, col]:
                v_count += 1
            else:
                break
        
        # Score calculation
        if h_count > 1 and v_count > 1:
            points = h_count + v_count
        elif h_count > 1:
            points = h_count
        elif v_count > 1:
            points = v_count
        else:
            points = 1
        
        return points
    
    def end_round_scoring(self) -> List[TileColor]:
        """
        Process end of round: move tiles to wall, calculate scores.
        Returns list of tiles to return to the box.
        """
        tiles_to_box = []
        
        # Process each pattern line
        for row in range(PATTERN_LINES):
            count, color = self.pattern_lines[row]
            max_capacity = row + 1
            
            if count == max_capacity and color is not None:
                # Line is complete - move one tile to wall
                wall_row, wall_col = get_wall_position(row, color)
                self.wall[wall_row, wall_col] = True
                
                # Score the placement
                self.score += self.score_tile_placement(wall_row, wall_col)
                
                # Remaining tiles go to box
                tiles_to_box.extend([color] * (count - 1))
                
                # Clear the pattern line
                self.pattern_lines[row] = (0, None)
        
        # Apply floor line penalties
        for i, tile in enumerate(self.floor_line):
            if i < len(FLOOR_PENALTIES):
                self.score += FLOOR_PENALTIES[i]
        
        # Ensure score doesn't go negative
        self.score = max(0, self.score)
        
        # First player marker goes back to center, regular tiles to box
        floor_tiles_to_box = [t for t in self.floor_line if t != TileColor.FIRST_PLAYER]
        tiles_to_box.extend(floor_tiles_to_box)
        
        has_first_player = TileColor.FIRST_PLAYER in self.floor_line
        self.floor_line = []
        
        return tiles_to_box, has_first_player
    
    def calculate_end_game_bonus(self) -> int:
        """Calculate end-game bonus points."""
        bonus = 0
        
        # Complete rows
        for row in range(WALL_SIZE):
            if all(self.wall[row, :]):
                bonus += BONUS_COMPLETE_ROW
        
        # Complete columns
        for col in range(WALL_SIZE):
            if all(self.wall[:, col]):
                bonus += BONUS_COMPLETE_COLUMN
        
        # Complete colors (all 5 tiles of one color placed)
        for color in range(NUM_TILE_COLORS):
            color_complete = True
            for row in range(WALL_SIZE):
                col = None
                for c, wall_color in enumerate(WALL_PATTERN[row]):
                    if wall_color == color:
                        col = c
                        break
                if col is None or not self.wall[row, col]:
                    color_complete = False
                    break
            if color_complete:
                bonus += BONUS_COMPLETE_COLOR
        
        return bonus
    
    def has_complete_row(self) -> bool:
        """Check if player has completed any horizontal row."""
        for row in range(WALL_SIZE):
            if all(self.wall[row, :]):
                return True
        return False
    
    def copy(self) -> 'PlayerBoard':
        """Create a deep copy of this board."""
        new_board = PlayerBoard()
        new_board.pattern_lines = self.pattern_lines.copy()
        new_board.wall = self.wall.copy()
        new_board.floor_line = self.floor_line.copy()
        new_board.score = self.score
        return new_board


class AzulGame:
    """
    Core Azul game logic.
    
    Manages game state, validates moves, and processes turns.
    """
    
    def __init__(self, num_players: int = 2, seed: Optional[int] = None):
        if num_players < 2 or num_players > 4:
            raise ValueError("Azul supports 2-4 players")
        
        self.num_players = num_players
        self.num_factories = FACTORIES_BY_PLAYERS[num_players]
        self.rng = random.Random(seed)
        
        # Initialize game state
        self.reset()
    
    def reset(self) -> None:
        """Reset game to initial state."""
        # Player boards
        self.player_boards: List[PlayerBoard] = [
            PlayerBoard() for _ in range(self.num_players)
        ]
        
        # Tile bag
        self.bag: List[TileColor] = []
        for color in range(NUM_TILE_COLORS):
            self.bag.extend([TileColor(color)] * TILES_PER_COLOR)
        self.rng.shuffle(self.bag)
        
        # Box lid (discarded tiles, refill bag when empty)
        self.box_lid: List[TileColor] = []
        
        # Factory displays: list of lists of tiles
        self.factories: List[List[TileColor]] = [
            [] for _ in range(self.num_factories)
        ]
        
        # Center of table
        self.center: List[TileColor] = []
        self.center_has_first_player: bool = False
        
        # Game state
        self.current_player: int = 0
        self.first_player_next_round: int = 0
        self.round_number: int = 1
        self.game_over: bool = False
        self.winner: Optional[int] = None
        
        # Start first round
        self._setup_round()
    
    def _setup_round(self) -> None:
        """Setup a new round by filling factories."""
        # Place first player marker in center
        self.center = []
        self.center_has_first_player = True
        
        # Fill each factory with 4 tiles
        for factory in self.factories:
            factory.clear()
            for _ in range(TILES_PER_FACTORY):
                tile = self._draw_tile()
                if tile is not None:
                    factory.append(tile)
    
    def _draw_tile(self) -> Optional[TileColor]:
        """Draw a tile from the bag. Refill from box lid if empty."""
        if not self.bag:
            if not self.box_lid:
                return None  # No tiles left
            self.bag = self.box_lid
            self.box_lid = []
            self.rng.shuffle(self.bag)
        
        return self.bag.pop() if self.bag else None
    
    def get_legal_actions(self) -> List[Tuple[int, TileColor, int]]:
        """
        Get all legal actions for current player.
        
        Returns list of (source, color, pattern_line) tuples where:
        - source: factory index (0 to num_factories-1) or -1 for center
        - color: TileColor to pick
        - pattern_line: row index (0-4) or -1 for floor
        """
        actions = []
        player_board = self.player_boards[self.current_player]
        
        # Check each factory
        for factory_idx, factory in enumerate(self.factories):
            if not factory:
                continue
            
            # Get unique colors in this factory
            colors = set(factory)
            for color in colors:
                # Can place on any valid pattern line
                for row in range(PATTERN_LINES):
                    if player_board.can_place_on_pattern_line(row, color):
                        actions.append((factory_idx, color, row))
                
                # Can always place directly on floor
                actions.append((factory_idx, color, -1))
        
        # Check center
        if self.center:
            colors = set(self.center)
            for color in colors:
                for row in range(PATTERN_LINES):
                    if player_board.can_place_on_pattern_line(row, color):
                        actions.append((-1, color, row))
                
                actions.append((-1, color, -1))
        
        return actions
    
    def take_action(self, action: Tuple[int, TileColor, int]) -> Dict[str, Any]:
        """
        Execute an action for the current player.
        
        Args:
            action: (source, color, pattern_line) tuple
        
        Returns:
            Dict with action results
        """
        source, color, pattern_line = action
        player_board = self.player_boards[self.current_player]
        result = {
            'player': self.current_player,
            'action': action,
            'tiles_taken': 0,
            'tiles_to_floor': 0,
            'took_first_player': False
        }
        
        # Get tiles from source
        if source == -1:
            # Taking from center
            tiles_taken = [t for t in self.center if t == color]
            self.center = [t for t in self.center if t != color]
            
            # Check for first player marker
            if self.center_has_first_player:
                self.center_has_first_player = False
                player_board.add_to_floor([TileColor.FIRST_PLAYER])
                self.first_player_next_round = self.current_player
                result['took_first_player'] = True
        else:
            # Taking from factory
            factory = self.factories[source]
            tiles_taken = [t for t in factory if t == color]
            remaining = [t for t in factory if t != color]
            
            # Move remaining to center
            self.center.extend(remaining)
            factory.clear()
        
        result['tiles_taken'] = len(tiles_taken)
        
        # Place tiles
        if pattern_line == -1:
            # All to floor
            player_board.add_to_floor(tiles_taken)
            result['tiles_to_floor'] = len(tiles_taken)
        else:
            # Place on pattern line, overflow goes to floor
            overflow = player_board.place_on_pattern_line(
                pattern_line, color, len(tiles_taken)
            )
            if overflow > 0:
                player_board.add_to_floor([color] * overflow)
                result['tiles_to_floor'] = overflow
        
        # Check if round ends (all factories and center empty)
        round_ended = self._check_round_end()
        result['round_ended'] = round_ended
        
        if round_ended:
            self._end_round()
            result['game_over'] = self.game_over
            if self.game_over:
                result['winner'] = self.winner
                result['final_scores'] = [pb.score for pb in self.player_boards]
        else:
            # Next player's turn
            self.current_player = (self.current_player + 1) % self.num_players
        
        return result
    
    def _check_round_end(self) -> bool:
        """Check if the round has ended (all tiles taken)."""
        if self.center or self.center_has_first_player:
            return False
        
        for factory in self.factories:
            if factory:
                return False
        
        return True
    
    def _end_round(self) -> None:
        """Process end of round scoring and setup next round."""
        # Process each player's board
        game_should_end = False
        
        for player_idx, player_board in enumerate(self.player_boards):
            tiles_to_box, had_first_player = player_board.end_round_scoring()
            self.box_lid.extend(tiles_to_box)
            
            # Check for game end condition
            if player_board.has_complete_row():
                game_should_end = True
        
        if game_should_end:
            self._end_game()
        else:
            # Setup next round
            self.round_number += 1
            self.current_player = self.first_player_next_round
            self._setup_round()
    
    def _end_game(self) -> None:
        """Process end of game scoring and determine winner."""
        self.game_over = True
        
        # Add end-game bonuses
        for player_board in self.player_boards:
            bonus = player_board.calculate_end_game_bonus()
            player_board.score += bonus
        
        # Determine winner (highest score, ties broken by complete rows)
        max_score = max(pb.score for pb in self.player_boards)
        winners = [i for i, pb in enumerate(self.player_boards) if pb.score == max_score]
        
        if len(winners) == 1:
            self.winner = winners[0]
        else:
            # Tiebreaker: most complete horizontal lines
            max_rows = -1
            for i in winners:
                complete_rows = sum(
                    1 for row in range(WALL_SIZE)
                    if all(self.player_boards[i].wall[row, :])
                )
                if complete_rows > max_rows:
                    max_rows = complete_rows
                    self.winner = i
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete game state as a dictionary."""
        return {
            'num_players': self.num_players,
            'current_player': self.current_player,
            'round_number': self.round_number,
            'game_over': self.game_over,
            'winner': self.winner,
            'factories': [list(f) for f in self.factories],
            'center': list(self.center),
            'center_has_first_player': self.center_has_first_player,
            'player_boards': [
                {
                    'pattern_lines': pb.pattern_lines.copy(),
                    'wall': pb.wall.copy(),
                    'floor_line': pb.floor_line.copy(),
                    'score': pb.score
                }
                for pb in self.player_boards
            ],
            'bag_size': len(self.bag),
            'box_lid_size': len(self.box_lid)
        }
    
    def copy(self) -> 'AzulGame':
        """Create a deep copy of the game state."""
        new_game = AzulGame.__new__(AzulGame)
        new_game.num_players = self.num_players
        new_game.num_factories = self.num_factories
        new_game.rng = random.Random()
        new_game.rng.setstate(self.rng.getstate())
        
        new_game.player_boards = [pb.copy() for pb in self.player_boards]
        new_game.bag = self.bag.copy()
        new_game.box_lid = self.box_lid.copy()
        new_game.factories = [f.copy() for f in self.factories]
        new_game.center = self.center.copy()
        new_game.center_has_first_player = self.center_has_first_player
        
        new_game.current_player = self.current_player
        new_game.first_player_next_round = self.first_player_next_round
        new_game.round_number = self.round_number
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        
        return new_game

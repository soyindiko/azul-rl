"""
PettingZoo environment for Azul board game.

This implements an AEC (Alternating Environment with Communication) environment
following PettingZoo conventions.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import functools

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gymnasium import spaces

from azul.game import AzulGame
from azul.constants import (
    TileColor, NUM_TILE_COLORS, PATTERN_LINES, WALL_SIZE,
    FLOOR_LINE_SIZE, FACTORIES_BY_PLAYERS, TILES_PER_FACTORY
)


class AzulEnv(AECEnv):
    """
    PettingZoo AEC environment for the Azul board game.
    
    Observation Space:
        A dictionary containing:
        - factories: (num_factories, num_colors) array of tile counts
        - center: (num_colors,) array of tile counts in center
        - center_has_first: bool indicating first player marker in center
        - my_pattern_lines: (5, 2) array of (count, color) for each line
        - my_wall: (5, 5) bool array
        - my_floor: (7,) array of floor tiles
        - my_score: int
        - opponent_walls: (num_opponents, 5, 5) bool arrays
        - opponent_scores: (num_opponents,) int array
    
    Action Space:
        Discrete space encoding (source, color, destination) tuples.
        - source: 0 to num_factories (last index = center)
        - color: 0 to 4 (tile colors)
        - destination: 0 to 5 (0-4 = pattern lines, 5 = floor)
    """
    
    metadata = {
        "name": "azul_v0",
        "render_modes": ["human", "ansi"],
        "is_parallelizable": False,
    }
    
    def __init__(
        self,
        num_players: int = 2,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None
    ):
        super().__init__()
        
        assert 2 <= num_players <= 4, "Azul supports 2-4 players"
        
        self.num_players = num_players
        self.num_factories = FACTORIES_BY_PLAYERS[num_players]
        self.render_mode = render_mode
        self._seed = seed
        
        # Create agent names
        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self.agent_name_mapping = {
            agent: i for i, agent in enumerate(self.possible_agents)
        }
        
        # Action space encoding
        # (num_factories + 1) sources * 5 colors * 6 destinations
        self.num_sources = self.num_factories + 1  # factories + center
        self.num_colors = NUM_TILE_COLORS
        self.num_destinations = PATTERN_LINES + 1  # pattern lines + floor
        
        self._action_space = spaces.Discrete(
            self.num_sources * self.num_colors * self.num_destinations
        )
        
        # Observation space
        self._observation_space = spaces.Dict({
            # Factory tiles: count of each color in each factory
            "factories": spaces.Box(
                low=0, high=TILES_PER_FACTORY,
                shape=(self.num_factories, NUM_TILE_COLORS),
                dtype=np.int8
            ),
            # Center tiles: count of each color
            "center": spaces.Box(
                low=0, high=100,  # Can accumulate many tiles
                shape=(NUM_TILE_COLORS,),
                dtype=np.int8
            ),
            # First player marker in center
            "center_has_first": spaces.Discrete(2),
            # My pattern lines: (count, color) for each - color is -1 if empty
            "my_pattern_lines": spaces.Box(
                low=-1, high=NUM_TILE_COLORS,
                shape=(PATTERN_LINES, 2),
                dtype=np.int8
            ),
            # My wall
            "my_wall": spaces.Box(
                low=0, high=1,
                shape=(WALL_SIZE, WALL_SIZE),
                dtype=np.int8
            ),
            # My floor line (tile colors, -1 for empty, 5 for first player marker)
            "my_floor": spaces.Box(
                low=-1, high=NUM_TILE_COLORS + 1,
                shape=(FLOOR_LINE_SIZE,),
                dtype=np.int8
            ),
            # My score
            "my_score": spaces.Box(low=0, high=500, shape=(1,), dtype=np.int32),
            # Opponent walls
            "opponent_walls": spaces.Box(
                low=0, high=1,
                shape=(num_players - 1, WALL_SIZE, WALL_SIZE),
                dtype=np.int8
            ),
            # Opponent scores
            "opponent_scores": spaces.Box(
                low=0, high=500,
                shape=(num_players - 1,),
                dtype=np.int32
            ),
            # Action mask for valid actions
            "action_mask": spaces.Box(
                low=0, high=1,
                shape=(self.num_sources * self.num_colors * self.num_destinations,),
                dtype=np.int8
            ),
        })
        
        self.game: Optional[AzulGame] = None
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        return self._observation_space
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        return self._action_space
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> None:
        """Reset the environment."""
        if seed is not None:
            self._seed = seed
        
        self.game = AzulGame(num_players=self.num_players, seed=self._seed)
        
        self.agents = self.possible_agents.copy()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
    
    def step(self, action: int) -> None:
        """Execute one step in the environment."""
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        
        agent = self.agent_selection
        player_idx = self.agent_name_mapping[agent]
        
        # Reset rewards
        for a in self.agents:
            self.rewards[a] = 0
        
        # Decode and validate action
        source, color, destination = self._decode_action(action)
        game_action = self._to_game_action(source, color, destination)
        
        # Check if action is legal
        legal_actions = self.game.get_legal_actions()
        
        if game_action not in legal_actions:
            # Invalid action penalty
            self.rewards[agent] = -10
            self._cumulative_rewards[agent] += self.rewards[agent]
            
            # Select a random legal action instead
            if legal_actions:
                game_action = legal_actions[0]
            else:
                # No legal actions (shouldn't happen in normal play)
                self.terminations = {a: True for a in self.agents}
                return
        
        # Execute action
        old_scores = [pb.score for pb in self.game.player_boards]
        result = self.game.take_action(game_action)
        new_scores = [pb.score for pb in self.game.player_boards]
        
        # Calculate rewards based on score changes
        for i, a in enumerate(self.agents):
            score_diff = new_scores[i] - old_scores[i]
            self.rewards[a] = score_diff
            self._cumulative_rewards[a] += self.rewards[a]
        
        # Check for game end
        if self.game.game_over:
            self.terminations = {a: True for a in self.agents}
            
            # Final reward: winner gets bonus
            if self.game.winner is not None:
                winner_agent = self.agents[self.game.winner]
                self.rewards[winner_agent] += 50
                self._cumulative_rewards[winner_agent] += 50
            
            self.infos = {
                a: {
                    "final_score": self.game.player_boards[self.agent_name_mapping[a]].score,
                    "winner": self.game.winner == self.agent_name_mapping[a]
                }
                for a in self.agents
            }
        
        # Move to next agent
        if not self.game.game_over:
            self.agent_selection = self._agent_selector.next()
            # Sync with game's current player
            while self.agent_name_mapping[self.agent_selection] != self.game.current_player:
                self.agent_selection = self._agent_selector.next()
    
    def observe(self, agent: str) -> Dict[str, np.ndarray]:
        """Get observation for the specified agent."""
        if self.game is None:
            return None
        
        player_idx = self.agent_name_mapping[agent]
        player_board = self.game.player_boards[player_idx]
        
        # Factory observation
        factories_obs = np.zeros((self.num_factories, NUM_TILE_COLORS), dtype=np.int8)
        for i, factory in enumerate(self.game.factories):
            for tile in factory:
                if tile < NUM_TILE_COLORS:
                    factories_obs[i, tile] += 1
        
        # Center observation
        center_obs = np.zeros(NUM_TILE_COLORS, dtype=np.int8)
        for tile in self.game.center:
            if tile < NUM_TILE_COLORS:
                center_obs[tile] += 1
        
        # Pattern lines
        pattern_obs = np.zeros((PATTERN_LINES, 2), dtype=np.int8)
        for i, (count, color) in enumerate(player_board.pattern_lines):
            pattern_obs[i, 0] = count
            pattern_obs[i, 1] = color if color is not None else -1
        
        # Wall
        wall_obs = player_board.wall.astype(np.int8)
        
        # Floor line
        floor_obs = np.full(FLOOR_LINE_SIZE, -1, dtype=np.int8)
        for i, tile in enumerate(player_board.floor_line):
            floor_obs[i] = tile
        
        # Opponent info
        opponent_idxs = [i for i in range(self.num_players) if i != player_idx]
        opponent_walls = np.stack([
            self.game.player_boards[i].wall.astype(np.int8)
            for i in opponent_idxs
        ])
        opponent_scores = np.array([
            self.game.player_boards[i].score
            for i in opponent_idxs
        ], dtype=np.int32)
        
        # Action mask
        action_mask = self._get_action_mask()
        
        return {
            "factories": factories_obs,
            "center": center_obs,
            "center_has_first": int(self.game.center_has_first_player),
            "my_pattern_lines": pattern_obs,
            "my_wall": wall_obs,
            "my_floor": floor_obs,
            "my_score": np.array([player_board.score], dtype=np.int32),
            "opponent_walls": opponent_walls,
            "opponent_scores": opponent_scores,
            "action_mask": action_mask,
        }
    
    def _encode_action(self, source: int, color: int, destination: int) -> int:
        """Encode (source, color, destination) tuple to discrete action."""
        return (
            source * (self.num_colors * self.num_destinations)
            + color * self.num_destinations
            + destination
        )
    
    def _decode_action(self, action: int) -> Tuple[int, int, int]:
        """Decode discrete action to (source, color, destination) tuple."""
        destination = action % self.num_destinations
        action //= self.num_destinations
        color = action % self.num_colors
        source = action // self.num_colors
        return source, color, destination
    
    def _to_game_action(
        self, source: int, color: int, destination: int
    ) -> Tuple[int, TileColor, int]:
        """Convert env action to game action format."""
        # source: 0 to num_factories-1 = factory, num_factories = center (-1 in game)
        game_source = source if source < self.num_factories else -1
        # destination: 0-4 = pattern line, 5 = floor (-1 in game)
        game_dest = destination if destination < PATTERN_LINES else -1
        return (game_source, TileColor(color), game_dest)
    
    def _from_game_action(
        self, game_action: Tuple[int, TileColor, int]
    ) -> Tuple[int, int, int]:
        """Convert game action to env action format."""
        source, color, dest = game_action
        env_source = source if source >= 0 else self.num_factories
        env_dest = dest if dest >= 0 else PATTERN_LINES
        return (env_source, int(color), env_dest)
    
    def _get_action_mask(self) -> np.ndarray:
        """Get mask of valid actions."""
        mask = np.zeros(
            self.num_sources * self.num_colors * self.num_destinations,
            dtype=np.int8
        )
        
        legal_actions = self.game.get_legal_actions()
        for game_action in legal_actions:
            env_action = self._from_game_action(game_action)
            encoded = self._encode_action(*env_action)
            mask[encoded] = 1
        
        return mask
    
    def render(self) -> Optional[str]:
        """Render the current game state."""
        if self.render_mode is None:
            return None
        
        if self.game is None:
            return "Game not initialized"
        
        output = []
        output.append(f"\n{'='*60}")
        output.append(f"AZUL - Round {self.game.round_number}")
        output.append(f"Current Player: {self.agent_selection}")
        output.append(f"{'='*60}\n")
        
        # Factories
        output.append("FACTORIES:")
        color_names = ['B', 'Y', 'R', 'K', 'W']  # Blue, Yellow, Red, blacK, White
        for i, factory in enumerate(self.game.factories):
            tiles = ''.join(color_names[t] for t in factory) if factory else "empty"
            output.append(f"  Factory {i}: [{tiles}]")
        
        # Center
        center_tiles = ''.join(color_names[t] for t in self.game.center if t < 5)
        first_marker = " (1st)" if self.game.center_has_first_player else ""
        output.append(f"\nCENTER: [{center_tiles}]{first_marker}\n")
        
        # Player boards
        for i, agent in enumerate(self.agents):
            pb = self.game.player_boards[i]
            marker = ">>>" if agent == self.agent_selection else "   "
            output.append(f"{marker} {agent} (Score: {pb.score})")
            
            # Pattern lines
            output.append("  Pattern Lines:")
            for row in range(PATTERN_LINES):
                count, color = pb.pattern_lines[row]
                color_char = color_names[color] if color is not None else '.'
                line = color_char * count + '.' * (row + 1 - count)
                output.append(f"    {row}: [{line}]")
            
            # Wall (simplified)
            output.append("  Wall:")
            for row in range(WALL_SIZE):
                wall_row = ''.join(
                    color_names[c] if pb.wall[row, col] else '.'
                    for col, c in enumerate([0,1,2,3,4])  # Simplified
                )
                output.append(f"    [{wall_row}]")
            
            # Floor
            floor_str = ''.join(
                '1' if t == TileColor.FIRST_PLAYER else color_names[t]
                for t in pb.floor_line
            )
            output.append(f"  Floor: [{floor_str}]")
            output.append("")
        
        result = '\n'.join(output)
        
        if self.render_mode == "human":
            print(result)
        
        return result
    
    def close(self) -> None:
        """Clean up resources."""
        pass

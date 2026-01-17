"""
Training script for Azul RL agent using MCTS.

This implements a simplified AlphaZero-style training loop:
1. Self-play: The agent plays games against itself using MCTS guided by the neural network
2. Data collection: Game states, MCTS policies, and outcomes are stored in a replay buffer
3. Training: The neural network learns to predict MCTS policies and game outcomes
4. Iteration: Repeat with the improved network

The key insight from AlphaZero is that MCTS provides "expert" move probabilities
that are better than the raw network output. By training the network to match
MCTS output, the network improves, which in turn improves MCTS, creating a
virtuous cycle of improvement.

Optimized for:
- Apple Silicon (M1/M2/M3/M4) with MPS (Metal Performance Shaders) support
- NVIDIA GPUs with CUDA and Automatic Mixed Precision (AMP) for Tensor Core acceleration
- Parallel self-play using multiple CPU cores

Mixed Precision (AMP):
- Enabled automatically on CUDA devices with Tensor Cores (T4, V100, A100, etc.)
- Uses FP16 for forward pass and FP32 for gradients
- Can provide 2-3x speedup on supported hardware with minimal accuracy loss

Usage:
    python train.py                           # Basic training with auto-detected settings
    python train.py --num-iterations 500      # Custom number of iterations
    python train.py --resume checkpoint.pt    # Resume from checkpoint
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Dict, Optional, Any
from collections import deque
import random
from tqdm import tqdm
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import json
from datetime import datetime

from azul.game import AzulGame
from azul.constants import (
    NUM_TILE_COLORS, PATTERN_LINES, WALL_SIZE, FLOOR_LINE_SIZE,
    FACTORIES_BY_PLAYERS
)
from mcts.mcts import MCTS, MCTSPlayer


# =============================================================================
# DEVICE DETECTION
# =============================================================================

def get_best_device() -> torch.device:
    """
    Auto-detect the best available compute device.
    
    Priority order:
    1. MPS (Metal Performance Shaders) - Apple Silicon GPU
    2. CUDA - NVIDIA GPU
    3. CPU - Fallback
    
    Returns:
        torch.device: The best available device for computation
    """
    if torch.backends.mps.is_available():
        # Apple Silicon (M1/M2/M3/M4) - use Metal GPU
        return torch.device("mps")
    elif torch.cuda.is_available():
        # NVIDIA GPU available
        return torch.device("cuda")
    else:
        # Fallback to CPU
        return torch.device("cpu")


def should_use_amp(device: torch.device) -> bool:
    """
    Determine if Automatic Mixed Precision (AMP) should be used.
    
    AMP uses FP16 for forward pass computations while keeping FP32 for
    gradient accumulation. This can provide significant speedups on
    GPUs with Tensor Cores (NVIDIA Volta, Turing, Ampere architectures).
    
    When to use AMP:
    - CUDA devices: Yes, especially T4, V100, A100, etc.
    - MPS (Apple Silicon): No, MPS has limited AMP support and can cause issues
    - CPU: No, AMP is designed for GPU acceleration
    
    Args:
        device: The compute device being used
    
    Returns:
        True if AMP should be enabled, False otherwise
    """
    if device.type == "cuda":
        # Check if CUDA supports AMP (requires compute capability >= 7.0 for best results)
        # but it works on older GPUs too, just without Tensor Core acceleration
        return True
    elif device.type == "mps":
        # MPS has experimental AMP support, but it's not stable
        # Better to skip for now on Apple Silicon
        return False
    else:
        # CPU - no benefit from AMP
        return False


def get_amp_context(use_amp: bool, device: torch.device):
    """
    Get the appropriate autocast context manager for mixed precision.
    
    Args:
        use_amp: Whether AMP is enabled
        device: The compute device
    
    Returns:
        Context manager for autocast (or nullcontext if AMP disabled)
    """
    if use_amp and device.type == "cuda":
        return torch.cuda.amp.autocast()
    else:
        # Return a no-op context manager
        from contextlib import nullcontext
        return nullcontext()


# =============================================================================
# NEURAL NETWORK
# =============================================================================

class AzulNet(nn.Module):
    """
    Neural network for Azul policy and value prediction.
    
    This is a dual-headed network similar to AlphaZero:
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Game State Input                         │
    │  (factories, center, pattern lines, wall, floor, scores)    │
    └─────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                   Shared Encoder                            │
    │        (4 layers of Linear → ReLU → LayerNorm)              │
    │                                                             │
    │  Learns a rich representation of the game state that        │
    │  captures strategic features useful for both policy         │
    │  and value prediction.                                      │
    └──────────────────┬──────────────────────┬───────────────────┘
                       │                      │
                       ▼                      ▼
    ┌──────────────────────────┐  ┌──────────────────────────────┐
    │      Policy Head         │  │        Value Head            │
    │                          │  │                              │
    │  Outputs probability     │  │  Outputs expected game       │
    │  distribution over all   │  │  outcome from this state     │
    │  possible actions        │  │  (-1 = lose, +1 = win)       │
    │                          │  │                              │
    │  Shape: (num_actions,)   │  │  Shape: (1,) with Tanh       │
    └──────────────────────────┘  └──────────────────────────────┘
    
    The policy head learns which moves are good.
    The value head learns who is winning from any position.
    
    Both heads share the encoder, which encourages the network to learn
    features that are useful for both tasks (multi-task learning).
    """
    
    def __init__(
        self,
        num_players: int = 2,
        hidden_size: int = 256,
        num_layers: int = 4
    ):
        """
        Initialize the neural network.
        
        Args:
            num_players: Number of players in the game (affects input/output size)
            hidden_size: Width of hidden layers (more = more capacity but slower)
            num_layers: Depth of encoder (more = can learn more complex patterns)
        """
        super().__init__()
        
        self.num_players = num_players
        self.num_factories = FACTORIES_BY_PLAYERS[num_players]
        
        # =================================================================
        # Calculate input feature size
        # We encode the entire game state as a flat vector of features
        # =================================================================
        
        # Factory displays: count of each color in each factory
        # Shape: (num_factories × 5 colors)
        factory_features = self.num_factories * NUM_TILE_COLORS
        
        # Center pool: count of each color + first player marker flag
        # Shape: (5 colors + 1 flag)
        center_features = NUM_TILE_COLORS + 1
        
        # Pattern lines: for each of 5 rows, store (tile_count, tile_color)
        # Shape: (5 rows × 2 values)
        pattern_features = PATTERN_LINES * 2
        
        # Wall: 5×5 binary grid showing which tiles are placed
        # Shape: (25,)
        wall_features = WALL_SIZE * WALL_SIZE
        
        # Floor line: 7 slots showing penalty tiles
        # Shape: (7,)
        floor_features = FLOOR_LINE_SIZE
        
        # Current score (normalized)
        # Shape: (1,)
        score_features = 1
        
        # Opponent information: their walls and scores
        # We need to see what opponents have to make good decisions
        opponent_wall_features = (num_players - 1) * WALL_SIZE * WALL_SIZE
        opponent_score_features = num_players - 1
        
        self.input_size = (
            factory_features + center_features + pattern_features +
            wall_features + floor_features + score_features +
            opponent_wall_features + opponent_score_features
        )
        
        # =================================================================
        # Calculate action space size
        # An action is: (source, color, destination)
        # - source: which factory (0 to N-1) or center (-1/N)
        # - color: which tile color (0 to 4)
        # - destination: which pattern line (0 to 4) or floor (-1/5)
        # =================================================================
        self.action_size = (self.num_factories + 1) * NUM_TILE_COLORS * (PATTERN_LINES + 1)
        
        # =================================================================
        # Build the shared encoder
        # We use LayerNorm instead of BatchNorm because:
        # 1. It works better with small batch sizes during inference
        # 2. It's faster on Apple Silicon
        # 3. It doesn't require tracking running statistics
        # =================================================================
        layers = []
        in_size = self.input_size
        for layer_idx in range(num_layers):
            layers.extend([
                nn.Linear(in_size, hidden_size),
                nn.ReLU(),  # Non-linearity allows learning complex patterns
                nn.LayerNorm(hidden_size)  # Normalizes activations for stable training
            ])
            in_size = hidden_size
        self.encoder = nn.Sequential(*layers)
        
        # =================================================================
        # Policy head: predicts action probabilities
        # Output is logits (pre-softmax) - we apply softmax during inference
        # =================================================================
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, self.action_size)
            # No activation - these are logits for cross-entropy loss
        )
        
        # =================================================================
        # Value head: predicts game outcome
        # Output is in [-1, 1] range via Tanh activation
        # -1 = certain loss, 0 = draw/uncertain, +1 = certain win
        # =================================================================
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Tanh()  # Squash output to [-1, 1] range
        )
    
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Batch of encoded game states, shape (batch_size, input_size)
        
        Returns:
            policy_logits: Action logits, shape (batch_size, action_size)
            value: State values, shape (batch_size, 1)
        """
        # Pass through shared encoder to get rich feature representation
        features = self.encoder(x)
        
        # Split into policy and value predictions
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        
        return policy_logits, value
    
    def predict(
        self, game_state: AzulGame, player: int
    ) -> Tuple[Dict[Tuple, float], float]:
        """
        Predict policy and value for a game state (inference mode).
        
        This method is used during MCTS to get the network's opinion on:
        1. Which actions are promising (policy)
        2. Who is likely to win from this state (value)
        
        Args:
            game_state: Current game state
            player: Which player's perspective to evaluate from
        
        Returns:
            policy: Dictionary mapping legal actions to their probabilities
            value: Expected outcome for this player (-1 to +1)
        """
        # Switch to evaluation mode (disables dropout, uses running stats for BatchNorm)
        self.eval()
        
        # No gradient computation needed for inference
        with torch.no_grad():
            # Encode the game state as a feature vector
            x = self.encode_state(game_state, player)
            x = x.unsqueeze(0)  # Add batch dimension: (input_size,) → (1, input_size)
            
            # Move to same device as model parameters
            device = next(self.parameters()).device
            x = x.to(device)
            
            # Get network predictions
            policy_logits, value = self(x)
            
            # Get list of legal actions (we only care about these)
            legal_actions = game_state.get_legal_actions()
            
            # Convert logits to probabilities via softmax
            # Move to CPU for easier manipulation
            policy = torch.softmax(policy_logits, dim=-1).squeeze().cpu()
            
            # Extract probabilities only for legal actions
            action_probs = {}
            for action in legal_actions:
                idx = self.action_to_index(action)
                action_probs[action] = policy[idx].item()
            
            # Renormalize so legal action probabilities sum to 1
            # (The network output includes illegal actions which we're ignoring)
            total = sum(action_probs.values())
            if total > 0:
                action_probs = {a: p / total for a, p in action_probs.items()}
            
            return action_probs, value.cpu().item()
    
    def encode_state(
        self, game_state: AzulGame, player: int
    ) -> torch.Tensor:
        """
        Encode a game state as a flat feature vector for the network.
        
        All features are normalized to roughly [0, 1] range for stable training.
        The encoding is from the perspective of the specified player.
        
        Args:
            game_state: The current game state to encode
            player: Which player's perspective (0 or 1 for 2-player game)
        
        Returns:
            Tensor of shape (input_size,) containing all features
        """
        features = []
        
        # -----------------------------------------------------------------
        # FACTORY FEATURES
        # Encode how many tiles of each color are in each factory
        # Normalized by 4 (max tiles per factory)
        # -----------------------------------------------------------------
        for factory in game_state.factories:
            factory_counts = np.zeros(NUM_TILE_COLORS, dtype=np.float32)
            for tile in factory:
                if tile < NUM_TILE_COLORS:  # Ignore special markers
                    factory_counts[tile] += 1
            features.extend(factory_counts / 4.0)  # Normalize by max factory size
        
        # -----------------------------------------------------------------
        # CENTER POOL FEATURES
        # Count of each color in the center, plus first player marker flag
        # Normalized by 20 (reasonable max for center accumulation)
        # -----------------------------------------------------------------
        center_counts = np.zeros(NUM_TILE_COLORS, dtype=np.float32)
        for tile in game_state.center:
            if tile < NUM_TILE_COLORS:
                center_counts[tile] += 1
        features.extend(center_counts / 20.0)  # Normalize
        features.append(float(game_state.center_has_first_player))  # Binary flag
        
        # -----------------------------------------------------------------
        # MY BOARD FEATURES
        # -----------------------------------------------------------------
        my_board = game_state.player_boards[player]
        
        # Pattern lines: encode (count, color) for each row
        # count normalized by 5 (max pattern line size)
        # color normalized by 5 (number of colors), -1 for empty
        for count, color in my_board.pattern_lines:
            features.append(count / 5.0)
            features.append((color if color is not None else -1) / 5.0)
        
        # Wall: binary 5×5 grid showing placed tiles (flattened)
        features.extend(my_board.wall.flatten().astype(np.float32))
        
        # Floor line: binary indicators for each slot
        # We just track if slots are filled, not which color
        floor = np.zeros(FLOOR_LINE_SIZE, dtype=np.float32)
        for i, tile in enumerate(my_board.floor_line):
            floor[i] = 1.0  # Slot is occupied
        features.extend(floor)
        
        # Score: normalized by 100 (typical max score)
        features.append(my_board.score / 100.0)
        
        # -----------------------------------------------------------------
        # OPPONENT FEATURES
        # We need to see opponent boards to make strategic decisions
        # (e.g., don't give them tiles they need)
        # -----------------------------------------------------------------
        for i in range(self.num_players):
            if i != player:
                opp_board = game_state.player_boards[i]
                # Opponent's wall (what they've built)
                features.extend(opp_board.wall.flatten().astype(np.float32))
                # Opponent's score
                features.append(opp_board.score / 100.0)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def action_to_index(self, action: Tuple[int, int, int]) -> int:
        """
        Convert a game action tuple to a network output index.
        
        Action format: (source, color, destination)
        - source: factory index (0 to N-1) or -1 for center
        - color: TileColor enum value (0 to 4)
        - destination: pattern line (0 to 4) or -1 for floor
        
        We map this to a single integer index for the policy vector.
        
        Args:
            action: (source, color, destination) tuple
        
        Returns:
            Integer index into the policy output vector
        """
        source, color, dest = action
        
        # Convert -1 (center) to num_factories for consistent indexing
        env_source = source if source >= 0 else self.num_factories
        
        # Convert -1 (floor) to PATTERN_LINES for consistent indexing
        env_dest = dest if dest >= 0 else PATTERN_LINES
        
        # Flatten to single index: source * (colors * dests) + color * dests + dest
        return (
            env_source * (NUM_TILE_COLORS * (PATTERN_LINES + 1)) +
            int(color) * (PATTERN_LINES + 1) +
            env_dest
        )
    
    def index_to_action(self, idx: int) -> Tuple[int, int, int]:
        """
        Convert a network output index back to a game action tuple.
        
        Inverse of action_to_index().
        
        Args:
            idx: Integer index from policy output
        
        Returns:
            (source, color, destination) action tuple
        """
        # Reverse the flattening
        dest = idx % (PATTERN_LINES + 1)
        idx //= (PATTERN_LINES + 1)
        color = idx % NUM_TILE_COLORS
        source = idx // NUM_TILE_COLORS
        
        # Convert back to game format (-1 for center/floor)
        game_source = source if source < self.num_factories else -1
        game_dest = dest if dest < PATTERN_LINES else -1
        
        return (game_source, color, game_dest)


# =============================================================================
# REPLAY BUFFER
# =============================================================================

class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling training data.
    
    Stores (state, policy, value) tuples from self-play games.
    Uses a fixed-size circular buffer that overwrites oldest experiences
    when full, ensuring we always train on relatively recent data.
    
    Why replay buffer?
    1. Breaks correlation between consecutive samples
    2. Allows reusing experiences multiple times
    3. Smooths out the training data distribution
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store.
                     Older experiences are overwritten when full.
        """
        # deque with maxlen automatically discards oldest items when full
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        state: torch.Tensor,
        policy: torch.Tensor,
        value: float
    ) -> None:
        """
        Add a single experience to the buffer.
        
        Args:
            state: Encoded game state tensor
            policy: MCTS policy distribution (target for policy head)
            value: Game outcome (target for value head)
        """
        self.buffer.append((state, policy, value))
    
    def push_batch(self, experiences: List[Tuple]) -> None:
        """
        Add multiple experiences to the buffer at once.
        
        Args:
            experiences: List of (state, policy, value) tuples
        """
        for exp in experiences:
            self.buffer.append(exp)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a random batch of experiences for training.
        
        Random sampling helps break correlations and provides
        a diverse training signal.
        
        Args:
            batch_size: Number of experiences to sample
        
        Returns:
            Tuple of (states, policies, values) tensors
        """
        # Sample random experiences (or all if buffer is smaller than batch)
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        # Stack into batched tensors
        states = torch.stack([x[0] for x in batch])
        policies = torch.stack([x[1] for x in batch])
        values = torch.tensor([x[2] for x in batch], dtype=torch.float32)
        
        return states, policies, values
    
    def __len__(self) -> int:
        """Return current number of experiences in buffer."""
        return len(self.buffer)


# =============================================================================
# PARALLEL SELF-PLAY WORKERS
# =============================================================================

# Global variables for worker processes
# These are initialized once per worker to avoid recreating the network
_worker_network = None
_worker_config = None


def _init_worker(network_state_dict: Dict, config: Dict) -> None:
    """
    Initialize a worker process with a copy of the neural network.
    
    Called once when each worker process starts. We create a fresh
    network instance and load the current weights.
    
    Args:
        network_state_dict: Network weights to load
        config: Configuration dictionary for self-play
    """
    global _worker_network, _worker_config
    _worker_config = config
    
    # Create network on CPU (workers don't use GPU to avoid contention)
    _worker_network = AzulNet(num_players=config['num_players'])
    _worker_network.load_state_dict(network_state_dict)
    _worker_network.eval()  # Set to evaluation mode


def _play_single_game(game_seed: int) -> List[Tuple]:
    """
    Play a single self-play game and return training data.
    
    This function runs in a worker process. It:
    1. Creates a new game with the given seed
    2. Uses MCTS (guided by the neural network) to select moves
    3. Records (state, MCTS_policy) pairs during play
    4. Assigns final values based on game outcome
    
    Args:
        game_seed: Random seed for game initialization (ensures variety)
    
    Returns:
        List of (state, policy, value) experiences from the game
    """
    global _worker_network, _worker_config
    
    config = _worker_config
    network = _worker_network
    
    # Create a new game with the specified seed for reproducibility
    game = AzulGame(num_players=config['num_players'], seed=game_seed)
    history = []  # Will store (state, policy, player) during game
    move_count = 0
    max_moves = 200  # Safety limit to prevent infinite games
    
    # -----------------------------------------------------------------
    # Create MCTS instance with neural network guidance
    # The network provides:
    # - policy_fn: Prior probabilities for which actions to explore
    # - value_fn: Estimated game outcome without playing to the end
    # -----------------------------------------------------------------
    def policy_fn(state):
        """Get action priors from neural network."""
        probs, _ = network.predict(state, state.current_player)
        return probs
    
    def value_fn(state):
        """Get state value estimate from neural network."""
        _, value = network.predict(state, state.current_player)
        return value
    
    mcts = MCTS(
        c_puct=config['c_puct'],            # Exploration constant
        num_simulations=config['num_simulations'],  # MCTS iterations per move
        policy_fn=policy_fn,
        value_fn=value_fn,
        use_rollouts=False  # Use network for leaf evaluation, not random rollouts
    )
    
    # -----------------------------------------------------------------
    # Play the game until completion or move limit
    # -----------------------------------------------------------------
    while not game.game_over and move_count < max_moves:
        current_player = game.current_player
        
        # Check for legal moves
        legal_actions = game.get_legal_actions()
        if not legal_actions:
            break  # No legal moves - game should end
        
        # Temperature controls exploration vs exploitation
        # High temperature (1.0) early = more exploration/variety
        # Low temperature (0) late = pick best move deterministically
        temp = config['temperature'] if move_count < config['temp_threshold'] else 0
        
        # Run MCTS to get improved action probabilities
        action_probs = mcts.get_action_probs(game, temperature=temp)
        
        # Fallback to uniform if MCTS returns empty (shouldn't happen)
        if not action_probs:
            action_probs = {a: 1.0 / len(legal_actions) for a in legal_actions}
        
        # -----------------------------------------------------------------
        # Store training data: (state, MCTS_policy)
        # The value will be filled in after the game ends
        # -----------------------------------------------------------------
        state_tensor = network.encode_state(game, current_player)
        policy_tensor = torch.zeros(network.action_size)
        for action, prob in action_probs.items():
            idx = network.action_to_index(action)
            policy_tensor[idx] = prob
        
        history.append((state_tensor, policy_tensor, current_player))
        
        # -----------------------------------------------------------------
        # Select action according to MCTS probabilities
        # -----------------------------------------------------------------
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        
        # Safety check for zero probabilities
        total_prob = sum(probs)
        if total_prob == 0:
            probs = [1.0 / len(actions)] * len(actions)
        
        # Sample action (with temperature already applied in get_action_probs)
        action = random.choices(actions, weights=probs)[0]
        game.take_action(action)
        move_count += 1
    
    # -----------------------------------------------------------------
    # Game finished: assign values based on outcome
    # Winner gets +1, loser gets -1, draw gets 0
    # -----------------------------------------------------------------
    experiences = []
    for state, policy, player in history:
        if game.winner == player:
            value = 1.0   # This player won
        elif game.winner is not None:
            value = -1.0  # This player lost
        else:
            value = 0.0   # Draw or game didn't finish properly
        experiences.append((state, policy, value))
    
    return experiences


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    """
    Training manager for Azul RL agent.
    
    Coordinates the AlphaZero training loop:
    1. Self-play: Generate games using current network + MCTS
    2. Training: Update network to match MCTS policies and game outcomes
    3. Repeat: The improved network makes MCTS stronger, which generates
               better training data, creating a virtuous cycle
    
    Optimizations for Apple Silicon:
    - Auto-detects MPS (Metal GPU) for fast training
    - Parallel self-play using multiple CPU cores
    - Efficient batch processing
    """
    
    def __init__(
        self,
        num_players: int = 2,
        num_simulations: int = 50,
        c_puct: float = 1.41,
        learning_rate: float = 0.001,
        batch_size: int = 128,
        buffer_size: int = 200000,
        device: str = "auto",
        num_workers: int = 0  # 0 = auto-detect
    ):
        """
        Initialize the trainer.
        
        Args:
            num_players: Number of players in Azul (2-4)
            num_simulations: MCTS simulations per move (more = stronger but slower)
            c_puct: MCTS exploration constant (higher = more exploration)
            learning_rate: Neural network learning rate
            batch_size: Training batch size (larger = more stable but needs more memory)
            buffer_size: Replay buffer capacity (larger = more diverse training data)
            device: Compute device ("auto", "cpu", "mps", "cuda")
            num_workers: Parallel self-play workers (0 = auto-detect)
        """
        self.num_players = num_players
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.batch_size = batch_size
        
        # -----------------------------------------------------------------
        # Device selection
        # MPS (Metal) provides significant speedup on Apple Silicon
        # -----------------------------------------------------------------
        if device == "auto":
            self.device = get_best_device()
        else:
            self.device = torch.device(device)
        
        print(f"🖥️  Using device: {self.device}")
        
        # -----------------------------------------------------------------
        # Worker pool for parallel self-play
        # We use half the CPU cores to leave room for OS and other processes
        # -----------------------------------------------------------------
        if num_workers == 0:
            self.num_workers = max(1, mp.cpu_count() // 2)
        else:
            self.num_workers = num_workers
        
        print(f"👷 Using {self.num_workers} parallel workers for self-play")
        
        # -----------------------------------------------------------------
        # Initialize neural network
        # -----------------------------------------------------------------
        self.network = AzulNet(num_players=num_players).to(self.device)
        
        # -----------------------------------------------------------------
        # Optimizer: AdamW with weight decay for regularization
        # Weight decay prevents overfitting by penalizing large weights
        # -----------------------------------------------------------------
        self.optimizer = optim.AdamW(
            self.network.parameters(),
            lr=learning_rate,
            weight_decay=1e-4  # L2 regularization
        )
        
        # -----------------------------------------------------------------
        # Learning rate scheduler: Cosine annealing with warm restarts
        # Periodically resets learning rate to escape local minima
        # T_0=50: First restart after 50 iterations
        # T_mult=2: Each subsequent cycle is 2x longer
        # -----------------------------------------------------------------
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=50,
            T_mult=2
        )
        
        # -----------------------------------------------------------------
        # Automatic Mixed Precision (AMP) setup
        # AMP uses FP16 for forward pass on CUDA to leverage Tensor Cores
        # GradScaler prevents gradient underflow in FP16
        # -----------------------------------------------------------------
        self.use_amp = should_use_amp(self.device)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print(f"⚡ Mixed Precision (AMP) enabled for faster training")
        else:
            self.scaler = None
            if self.device.type == "mps":
                print(f"💻 MPS device detected - using full precision (AMP not recommended for MPS)")
            elif self.device.type == "cpu":
                print(f"💻 CPU device - AMP not applicable")
        
        # -----------------------------------------------------------------
        # Replay buffer for storing self-play experiences
        # -----------------------------------------------------------------
        self.buffer = ReplayBuffer(capacity=buffer_size)
        
        # -----------------------------------------------------------------
        # Training statistics
        # -----------------------------------------------------------------
        self.train_steps = 0     # Total gradient updates
        self.games_played = 0    # Total self-play games
        self.iteration = 0       # Current training iteration
        
        # -----------------------------------------------------------------
        # Training history for logging and visualization
        # Stores metrics from each iteration for later analysis
        # -----------------------------------------------------------------
        self.training_history = []
    
    def self_play(
        self,
        num_games: int = 10,
        temperature: float = 1.0,
        temperature_threshold: int = 30
    ) -> int:
        """
        Generate self-play games to collect training data.
        
        Uses parallel workers for efficiency on multi-core systems.
        
        Args:
            num_games: Number of games to play
            temperature: MCTS temperature for action selection (1.0 = proportional,
                        0 = greedy). Higher values increase exploration.
            temperature_threshold: After this many moves, switch to greedy (temp=0)
        
        Returns:
            Total number of experiences (state/policy/value tuples) generated
        """
        # Configuration for worker processes
        config = {
            'num_players': self.num_players,
            'num_simulations': self.num_simulations,
            'c_puct': self.c_puct,
            'temperature': temperature,
            'temp_threshold': temperature_threshold
        }
        
        # -----------------------------------------------------------------
        # Get network weights for workers
        # We need to copy to CPU since workers run on CPU
        # -----------------------------------------------------------------
        network_state = self.network.cpu().state_dict()
        self.network.to(self.device)  # Move back to training device
        
        total_experiences = 0
        
        # -----------------------------------------------------------------
        # Parallel self-play (when we have enough games and workers)
        # -----------------------------------------------------------------
        if self.num_workers > 1 and num_games >= self.num_workers:
            # Generate random seeds for game variety
            game_seeds = [random.randint(0, 2**31) for _ in range(num_games)]
            
            # Create worker pool and submit games
            with ProcessPoolExecutor(
                max_workers=self.num_workers,
                initializer=_init_worker,
                initargs=(network_state, config)
            ) as executor:
                # Submit all games to the worker pool
                futures = [
                    executor.submit(_play_single_game, seed)
                    for seed in game_seeds
                ]
                
                # Collect results as they complete
                for future in tqdm(
                    as_completed(futures),
                    total=num_games,
                    desc="Self-play"
                ):
                    experiences = future.result()
                    self.buffer.push_batch(experiences)
                    total_experiences += len(experiences)
                    self.games_played += 1
        else:
            # -----------------------------------------------------------------
            # Sequential self-play (for debugging or small batches)
            # -----------------------------------------------------------------
            for _ in tqdm(range(num_games), desc="Self-play"):
                experiences = self._play_single_game_sequential(config)
                self.buffer.push_batch(experiences)
                total_experiences += len(experiences)
                self.games_played += 1
        
        return total_experiences
    
    def _play_single_game_sequential(self, config: Dict) -> List[Tuple]:
        """
        Play a single game sequentially (on main process).
        
        Used when parallel execution isn't beneficial (debugging, small batches).
        Same logic as the parallel worker function.
        
        Args:
            config: Game configuration dictionary
        
        Returns:
            List of (state, policy, value) experiences
        """
        game = AzulGame(num_players=config['num_players'])
        history = []
        move_count = 0
        max_moves = 200
        
        # MCTS with neural network guidance
        def policy_fn(state):
            probs, _ = self.network.predict(state, state.current_player)
            return probs
        
        def value_fn(state):
            _, value = self.network.predict(state, state.current_player)
            return value
        
        mcts = MCTS(
            c_puct=config['c_puct'],
            num_simulations=config['num_simulations'],
            policy_fn=policy_fn,
            value_fn=value_fn,
            use_rollouts=False
        )
        
        # Play until game over or move limit
        while not game.game_over and move_count < max_moves:
            current_player = game.current_player
            
            legal_actions = game.get_legal_actions()
            if not legal_actions:
                break
            
            # Temperature annealing
            temp = config['temperature'] if move_count < config['temp_threshold'] else 0
            action_probs = mcts.get_action_probs(game, temperature=temp)
            
            if not action_probs:
                action_probs = {a: 1.0 / len(legal_actions) for a in legal_actions}
            
            # Record state and policy
            state_tensor = self.network.encode_state(game, current_player)
            policy_tensor = torch.zeros(self.network.action_size)
            for action, prob in action_probs.items():
                idx = self.network.action_to_index(action)
                policy_tensor[idx] = prob
            
            history.append((state_tensor, policy_tensor, current_player))
            
            # Select and execute action
            actions = list(action_probs.keys())
            probs = list(action_probs.values())
            
            total_prob = sum(probs)
            if total_prob == 0:
                probs = [1.0 / len(actions)] * len(actions)
            
            action = random.choices(actions, weights=probs)[0]
            game.take_action(action)
            move_count += 1
        
        # Assign values based on game outcome
        experiences = []
        for state, policy, player in history:
            if game.winner == player:
                value = 1.0
            elif game.winner is not None:
                value = -1.0
            else:
                value = 0.0
            experiences.append((state, policy, value))
        
        return experiences
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step (forward + backward pass).
        
        Samples a batch from the replay buffer and updates the network to:
        1. Match MCTS policies (cross-entropy loss)
        2. Predict game outcomes (MSE loss)
        
        Uses Automatic Mixed Precision (AMP) on CUDA devices for faster training.
        AMP keeps FP32 precision for gradient accumulation while using FP16
        for forward pass, leveraging Tensor Cores on supported GPUs.
        
        Returns:
            Dictionary with loss values for logging
        """
        # Skip if not enough data
        if len(self.buffer) < self.batch_size:
            return {"policy_loss": 0, "value_loss": 0, "total_loss": 0}
        
        # Set network to training mode (enables dropout, etc.)
        self.network.train()
        
        # -----------------------------------------------------------------
        # Sample batch from replay buffer
        # -----------------------------------------------------------------
        states, target_policies, target_values = self.buffer.sample(self.batch_size)
        
        # Move to training device
        states = states.to(self.device)
        target_policies = target_policies.to(self.device)
        target_values = target_values.to(self.device)
        
        # Clear gradients from previous step
        self.optimizer.zero_grad()
        
        # -----------------------------------------------------------------
        # Forward pass (with optional Mixed Precision)
        # autocast automatically converts operations to FP16 where safe
        # -----------------------------------------------------------------
        with get_amp_context(self.use_amp, self.device):
            policy_logits, values = self.network(states)
            
            # -----------------------------------------------------------------
            # Compute losses
            # -----------------------------------------------------------------
            
            # Policy loss: cross-entropy between network output and MCTS policy
            # This teaches the network to predict what MCTS would choose
            policy_loss = F.cross_entropy(policy_logits, target_policies)
            
            # Value loss: MSE between predicted and actual game outcome
            # This teaches the network to evaluate positions accurately
            value_loss = F.mse_loss(values.squeeze(), target_values)
            
            # Combined loss (equal weighting of both objectives)
            total_loss = policy_loss + value_loss
        
        # -----------------------------------------------------------------
        # Backward pass and optimization
        # With AMP: GradScaler prevents gradient underflow in FP16
        # Without AMP: Standard backward pass
        # -----------------------------------------------------------------
        if self.use_amp and self.scaler is not None:
            # Scale loss to prevent FP16 underflow, then backward
            self.scaler.scale(total_loss).backward()
            
            # Unscale gradients before clipping
            self.scaler.unscale_(self.optimizer)
            
            # Gradient clipping prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            
            # Optimizer step with scaler (skips if gradients contain inf/nan)
            self.scaler.step(self.optimizer)
            
            # Update scaler for next iteration
            self.scaler.update()
        else:
            # Standard backward pass (CPU or MPS)
            total_loss.backward()
            
            # Gradient clipping prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
        
        self.train_steps += 1
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item()
        }
    
    def train(
        self,
        num_iterations: int = 1000,
        games_per_iteration: int = 20,
        train_steps_per_iteration: int = 200,
        save_interval: int = 10,
        save_path: str = "checkpoints"
    ) -> None:
        """
        Main training loop.
        
        Each iteration:
        1. Generates self-play games with current network
        2. Trains network on accumulated experiences
        3. Periodically saves checkpoints
        
        Args:
            num_iterations: Total training iterations
            games_per_iteration: Self-play games per iteration
            train_steps_per_iteration: Gradient updates per iteration
            save_interval: Save checkpoint every N iterations
            save_path: Directory for saving checkpoints
        """
        # Create checkpoint directory
        os.makedirs(save_path, exist_ok=True)
        
        # -----------------------------------------------------------------
        # Calculate starting iteration for resume support
        # If resuming, self.iteration contains the last completed iteration
        # We start from the next one
        # -----------------------------------------------------------------
        start_iteration = self.iteration + 1 if self.iteration > 0 else 0
        end_iteration = start_iteration + num_iterations
        
        # Print training configuration
        print(f"\n🎮 Starting Azul RL Training")
        print(f"{'='*60}")
        if start_iteration > 0:
            print(f"   📂 Resuming from iteration {start_iteration}")
        print(f"   Iterations:        {start_iteration} → {end_iteration} ({num_iterations} new)")
        print(f"   Games/iteration:   {games_per_iteration}")
        print(f"   Train steps/iter:  {train_steps_per_iteration}")
        print(f"   MCTS simulations:  {self.num_simulations}")
        print(f"   Batch size:        {self.batch_size}")
        print(f"   Device:            {self.device}")
        print(f"   Workers:           {self.num_workers}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # -----------------------------------------------------------------
        # Main training loop
        # Uses global iteration count for checkpoints and logging
        # -----------------------------------------------------------------
        for i in range(num_iterations):
            # Calculate the global iteration number
            global_iteration = start_iteration + i
            self.iteration = global_iteration
            iter_start = time.time()
            
            print(f"\n{'='*60}")
            print(f"=== Iteration {global_iteration + 1}/{end_iteration} (this session: {i + 1}/{num_iterations}) ===")
            print(f"{'='*60}")
            
            # -------------------------------------------------------------
            # Phase 1: Self-play
            # Generate training data by playing games with MCTS
            # -------------------------------------------------------------
            num_exp = self.self_play(num_games=games_per_iteration)
            print(f"📊 Generated {num_exp} experiences, Buffer size: {len(self.buffer)}")
            
            # -------------------------------------------------------------
            # Phase 2: Training
            # Update network to better match MCTS
            # -------------------------------------------------------------
            losses = []
            for _ in tqdm(range(train_steps_per_iteration), desc="Training"):
                loss = self.train_step()
                losses.append(loss)
            
            # Update learning rate according to schedule
            self.scheduler.step()
            
            # Calculate average losses
            avg_loss = {
                k: np.mean([l[k] for l in losses])
                for k in losses[0].keys()
            }
            
            # -------------------------------------------------------------
            # Logging
            # -------------------------------------------------------------
            iter_time = time.time() - iter_start
            total_time = time.time() - start_time
            current_lr = self.scheduler.get_last_lr()[0]
            
            print(f"\n📈 Losses:")
            print(f"   Policy loss: {avg_loss['policy_loss']:.4f}")
            print(f"   Value loss:  {avg_loss['value_loss']:.4f}")
            print(f"   Total loss:  {avg_loss['total_loss']:.4f}")
            print(f"\n⏱️  Iteration time: {iter_time:.1f}s")
            print(f"   Total time: {total_time/60:.1f} minutes")
            print(f"🎯 Games played: {self.games_played}")
            print(f"📚 Learning rate: {current_lr:.6f}")
            
            # -------------------------------------------------------------
            # Record training history for later visualization
            # -------------------------------------------------------------
            history_entry = {
                "iteration": global_iteration + 1,
                "timestamp": datetime.now().isoformat(),
                "policy_loss": float(avg_loss['policy_loss']),
                "value_loss": float(avg_loss['value_loss']),
                "total_loss": float(avg_loss['total_loss']),
                "learning_rate": float(current_lr),
                "games_played": self.games_played,
                "train_steps": self.train_steps,
                "buffer_size": len(self.buffer),
                "experiences_generated": num_exp,
                "iteration_time": iter_time,
                "total_time": total_time
            }
            self.training_history.append(history_entry)
            
            # -------------------------------------------------------------
            # Save checkpoint periodically
            # Uses global iteration number so checkpoints are sequential
            # even when resuming from a previous training run
            # -------------------------------------------------------------
            if (global_iteration + 1) % save_interval == 0:
                path = f"{save_path}/model_iter_{global_iteration + 1}.pt"
                self.save(path)
                print(f"\n💾 Saved checkpoint: {path}")
                
                # Save training history alongside checkpoint
                self.save_training_history(f"{save_path}/training_history.json")
        
        # Save final model with global iteration number
        final_iteration = start_iteration + num_iterations
        final_path = f"{save_path}/model_iter_{final_iteration}.pt"
        self.save(final_path)
        print(f"\n{'='*60}")
        print(f"✅ Training complete!")
        print(f"   Total iterations: {final_iteration}")
        print(f"   Total games: {self.games_played}")
        print(f"   Total train steps: {self.train_steps}")
        print(f"   Final model: {final_path}")
        print(f"{'='*60}")
        
        # Save final training history
        self.save_training_history(f"{save_path}/training_history.json")
    
    def save_training_history(self, path: str) -> None:
        """
        Save training history to a JSON file for visualization.
        
        The history contains per-iteration metrics:
        - Losses (policy, value, total)
        - Learning rate
        - Games played, train steps
        - Timing information
        
        If the file already exists, it merges the new history with existing data,
        avoiding duplicates based on iteration number.
        
        Args:
            path: File path for the JSON history file
        """
        # Load existing history if file exists
        existing_history = []
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    existing_history = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing_history = []
        
        # Merge histories, avoiding duplicates by iteration number
        existing_iterations = {h['iteration'] for h in existing_history}
        merged_history = existing_history.copy()
        
        for entry in self.training_history:
            if entry['iteration'] not in existing_iterations:
                merged_history.append(entry)
        
        # Sort by iteration
        merged_history.sort(key=lambda x: x['iteration'])
        
        # Save to file
        with open(path, 'w') as f:
            json.dump(merged_history, f, indent=2)
        
        print(f"📊 Training history saved to: {path} ({len(merged_history)} entries)")
    
    def load_training_history(self, path: str) -> None:
        """
        Load training history from a JSON file.
        
        Called when resuming training to continue logging from where we left off.
        
        Args:
            path: File path to the JSON history file
        """
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    self.training_history = json.load(f)
                print(f"📊 Loaded training history: {len(self.training_history)} entries")
            except (json.JSONDecodeError, IOError):
                print(f"⚠️ Could not load training history, starting fresh")
                self.training_history = []
        else:
            self.training_history = []
    
    def save(self, path: str) -> None:
        """
        Save model checkpoint to disk.
        
        Saves everything needed to resume training or use the model:
        - Network weights
        - Optimizer state (momentum, etc.)
        - Scheduler state
        - GradScaler state (if using AMP)
        - Training statistics
        
        Args:
            path: File path for the checkpoint
        """
        checkpoint = {
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_steps": self.train_steps,
            "games_played": self.games_played,
            "iteration": self.iteration,
            "use_amp": self.use_amp  # Save AMP state for reference
        }
        
        # Save GradScaler state if using AMP
        if self.use_amp and self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load(self, path: str) -> None:
        """
        Load model checkpoint from disk.
        
        Restores all state needed to continue training.
        Handles AMP/GradScaler state intelligently:
        - If checkpoint was saved with AMP and we're using AMP: restore scaler state
        - If checkpoint was saved without AMP but we're using AMP: initialize fresh scaler
        - If checkpoint was saved with AMP but we're not using AMP: ignore scaler state
        
        Args:
            path: File path to the checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.train_steps = checkpoint["train_steps"]
        self.games_played = checkpoint["games_played"]
        self.iteration = checkpoint.get("iteration", 0)
        
        # Restore GradScaler state if available and we're using AMP
        if self.use_amp and self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            print(f"⚡ Restored AMP GradScaler state")
        elif self.use_amp and "scaler_state_dict" not in checkpoint:
            print(f"⚡ AMP enabled but checkpoint has no scaler state (starting fresh)")
        
        # Try to load training history from the same directory
        checkpoint_dir = os.path.dirname(path)
        history_path = os.path.join(checkpoint_dir, "training_history.json")
        self.load_training_history(history_path)
        
        print(f"📂 Loaded checkpoint from iteration {self.iteration + 1}")
        print(f"   Games played: {self.games_played}")
        print(f"   Train steps: {self.train_steps}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main entry point for training.
    
    Parses command-line arguments and starts the training loop.
    
    Usage examples:
        python train.py                                    # Default settings
        python train.py --num-iterations 500               # More iterations
        python train.py --device mps --workers 8           # Custom hardware
        python train.py --resume checkpoints/model.pt      # Resume training
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train an Azul RL agent using AlphaZero-style MCTS + neural network"
    )
    
    # Game settings
    parser.add_argument("--num-players", type=int, default=2,
                       help="Number of players in the game (default: 2)")
    
    # Training loop settings
    parser.add_argument("--num-iterations", type=int, default=1000,
                       help="Number of training iterations (default: 1000)")
    parser.add_argument("--games-per-iteration", type=int, default=20,
                       help="Self-play games per iteration (default: 20)")
    parser.add_argument("--train-steps", type=int, default=200,
                       help="Gradient updates per iteration (default: 200)")
    
    # MCTS settings
    parser.add_argument("--num-simulations", type=int, default=50,
                       help="MCTS simulations per move (default: 50)")
    
    # Neural network settings
    parser.add_argument("--batch-size", type=int, default=128,
                       help="Training batch size (default: 128)")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate (default: 0.001)")
    
    # Checkpointing
    parser.add_argument("--save-path", type=str, default="checkpoints",
                       help="Directory for saving checkpoints (default: checkpoints)")
    parser.add_argument("--save-interval", type=int, default=10,
                       help="Save checkpoint every N iterations (default: 10)")
    
    # Hardware settings
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "mps", "cuda"],
                       help="Compute device (default: auto-detect)")
    parser.add_argument("--workers", type=int, default=0,
                       help="Parallel self-play workers (0=auto, default: 0)")
    
    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume training from checkpoint file")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = Trainer(
        num_players=args.num_players,
        num_simulations=args.num_simulations,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.workers
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load(args.resume)
    
    # Start training
    trainer.train(
        num_iterations=args.num_iterations,
        games_per_iteration=args.games_per_iteration,
        train_steps_per_iteration=args.train_steps,
        save_interval=args.save_interval,
        save_path=args.save_path
    )


if __name__ == "__main__":
    main()

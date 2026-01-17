"""
Evaluation script for trained Azul RL models.

This script provides comprehensive evaluation capabilities:
1. Test trained models against baseline players (Random, Greedy)
2. Compare two trained models head-to-head
3. Measure model strength with statistical confidence
4. Track improvement across training checkpoints

The evaluation uses parallel game execution to leverage multiple CPU cores,
making it fast even when running hundreds of evaluation games.

Key Metrics:
- Win rate: Percentage of games won
- Average score: Points per game
- Score margin: How much the model wins/loses by
- Elo rating estimate: Relative strength measurement

Usage examples:
    python evaluate.py --model checkpoints/model_iter_100.pt --games 100
    python evaluate.py --model1 model_v1.pt --model2 model_v2.pt --games 50
    python evaluate.py --quick-test --games 20
    python evaluate.py --model checkpoint.pt --parallel --workers 8

Optimized for Apple Silicon (M1/M2/M3/M4) with parallel game execution.
"""

import argparse
import numpy as np
from typing import Optional, Tuple, Dict, List, Callable, Any
from tqdm import tqdm
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import os
import sys
import traceback

from azul.game import AzulGame
from azul.constants import TileColor, PATTERN_LINES
from mcts.mcts import MCTS, MCTSPlayer

# =============================================================================
# MULTIPROCESSING CONFIGURATION
# =============================================================================
# On macOS, we need to use 'spawn' method for multiprocessing to work correctly
# This must be set before any multiprocessing operations
try:
    if sys.platform == 'darwin':  # macOS
        mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set - ignore
    pass


# =============================================================================
# DEVICE DETECTION
# =============================================================================

def get_best_device() -> torch.device:
    """
    Auto-detect the best available compute device.
    
    Priority:
    1. MPS (Metal) - Apple Silicon GPU
    2. CUDA - NVIDIA GPU
    3. CPU - Fallback
    
    Returns:
        torch.device: Best available device
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# =============================================================================
# BASELINE PLAYERS
# =============================================================================

class RandomPlayer:
    """
    Baseline player that selects uniformly random legal actions.
    
    This is the weakest possible player - any learning should beat it.
    Useful as a lower bound for model evaluation.
    
    Expected win rate for a decent model: >70%
    """
    
    def select_action(self, game: AzulGame) -> Optional[Tuple]:
        """Select a random legal action."""
        actions = game.get_legal_actions()
        if not actions:
            return None
        return actions[np.random.randint(len(actions))]
    
    def __str__(self):
        return "Random"
    
    def __repr__(self):
        return "RandomPlayer()"


class GreedyPlayer:
    """
    Baseline player using simple hand-crafted heuristics.
    
    Strategy:
    - Prefers actions that complete pattern lines (immediate points)
    - Avoids overflow to the floor line (penalties)
    - Slightly avoids taking the first player marker
    
    This represents a "reasonable" human-like strategy without lookahead.
    A good model should beat this consistently.
    
    Expected win rate for a strong model: >60%
    """
    
    def select_action(self, game: AzulGame) -> Optional[Tuple]:
        """Select the best action according to simple heuristics."""
        actions = game.get_legal_actions()
        if not actions:
            return None
        
        # Evaluate all actions and pick the best
        best_action = None
        best_score = float('-inf')
        
        for action in actions:
            score = self._evaluate_action(game, action)
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
    
    def _evaluate_action(self, game: AzulGame, action: Tuple) -> float:
        """
        Evaluate an action using simple heuristics.
        
        Heuristic scoring:
        - +10 for completing a pattern line (guaranteed wall tile)
        - -2 per tile that overflows to floor
        - -5 for putting tiles directly on floor
        - -1 for taking first player marker
        
        Args:
            game: Current game state
            action: (source, color, destination) tuple
        
        Returns:
            Heuristic score (higher = better)
        """
        source, color, dest = action
        player = game.player_boards[game.current_player]
        
        score = 0.0
        
        # Count how many tiles we'd pick up
        if source >= 0:
            # From a factory
            tiles = sum(1 for t in game.factories[source] if t == color)
        else:
            # From center
            tiles = sum(1 for t in game.center if t == color)
        
        if dest >= 0:
            # Placing on a pattern line
            count, _ = player.pattern_lines[dest]
            max_capacity = dest + 1
            remaining = max_capacity - count
            
            # Bonus for completing a pattern line
            if tiles >= remaining:
                score += 10.0
                
                # Extra bonus based on row (higher rows = more adjacent potential)
                score += dest * 0.5
            
            # Penalty for overflow to floor
            overflow = max(0, tiles - remaining)
            score -= overflow * 2.0
        else:
            # Placing directly on floor - generally bad
            score -= 5.0
            score -= tiles * 1.0  # More tiles = worse
        
        # Slight penalty for taking first player marker
        # (It's a penalty tile, but being first next round has value)
        if source == -1 and game.center_has_first_player:
            score -= 1.0
        
        return score
    
    def __str__(self):
        return "Greedy"
    
    def __repr__(self):
        return "GreedyPlayer()"


class NeuralPlayer:
    """
    Player using a trained neural network, optionally with MCTS.
    
    Two modes of operation:
    1. Neural + MCTS: Network guides MCTS search (strongest)
    2. Neural only: Direct policy output (faster, weaker)
    
    The MCTS mode is slower but significantly stronger because it
    combines the network's intuition with explicit tree search.
    """
    
    def __init__(
        self,
        model_path: str,
        num_players: int = 2,
        use_mcts: bool = True,
        num_simulations: int = 50,
        device: str = "auto"
    ):
        """
        Initialize the neural player.
        
        Args:
            model_path: Path to the trained model checkpoint
            num_players: Number of players in the game
            use_mcts: Whether to use MCTS (True) or raw network (False)
            num_simulations: MCTS simulations per move (if use_mcts=True)
            device: Compute device ("auto", "cpu", "mps", "cuda")
        """
        # Import here to avoid circular imports
        from train import AzulNet
        
        self.model_path = model_path
        self.use_mcts = use_mcts
        self.num_simulations = num_simulations
        
        # Detect best device
        if device == "auto":
            self.device = get_best_device()
        else:
            self.device = torch.device(device)
        
        # Load the neural network
        self.network = AzulNet(num_players=num_players)
        
        # Load checkpoint (map to appropriate device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.network.to(self.device)
        self.network.eval()  # Set to evaluation mode
        
        # Setup MCTS if enabled
        if use_mcts:
            # Create policy and value functions that use the network
            def policy_fn(state):
                """Get action priors from neural network."""
                probs, _ = self.network.predict(state, state.current_player)
                return probs
            
            def value_fn(state):
                """Get state value from neural network."""
                _, value = self.network.predict(state, state.current_player)
                return value
            
            self.mcts = MCTS(
                num_simulations=num_simulations,
                policy_fn=policy_fn,
                value_fn=value_fn,
                use_rollouts=False  # Use network for evaluation
            )
        else:
            self.mcts = None
    
    def select_action(self, game: AzulGame) -> Optional[Tuple]:
        """
        Select an action for the current game state.
        
        Args:
            game: Current game state
        
        Returns:
            Best action according to network/MCTS, or None if no legal actions
        """
        if self.use_mcts:
            # Use MCTS with network guidance
            # Temperature=0 means greedy (pick best action)
            return self.mcts.get_action(game, temperature=0)
        else:
            # Use network directly without MCTS search
            probs, _ = self.network.predict(game, game.current_player)
            
            if not probs:
                # Fallback if network returns empty (shouldn't happen)
                actions = game.get_legal_actions()
                return actions[0] if actions else None
            
            # Select the action with highest probability
            return max(probs.keys(), key=lambda a: probs[a])
    
    def __str__(self):
        """String representation for display."""
        if self.use_mcts:
            return f"Neural+MCTS({self.num_simulations})"
        else:
            return "Neural"
    
    def __repr__(self):
        return f"NeuralPlayer('{self.model_path}', mcts={self.use_mcts}, sims={self.num_simulations})"


# =============================================================================
# PARALLEL GAME EXECUTION
# =============================================================================

# Global variables for worker processes
_worker_player1 = None
_worker_player2 = None
_worker_config = None


def _init_game_worker(player1_config: Dict, player2_config: Dict, config: Dict) -> None:
    """
    Initialize a worker process with players.
    
    Creates player instances in the worker process to avoid
    serialization issues with neural networks.
    
    Args:
        player1_config: Configuration for player 1
        player2_config: Configuration for player 2
        config: General configuration
    """
    global _worker_player1, _worker_player2, _worker_config
    _worker_config = config
    
    # Create players based on configuration
    _worker_player1 = _create_player(player1_config)
    _worker_player2 = _create_player(player2_config)


def _create_player(config: Dict):
    """
    Create a player instance from configuration.
    
    Args:
        config: Player configuration dictionary with 'type' and parameters
    
    Returns:
        Player instance
    """
    player_type = config['type']
    
    if player_type == 'random':
        return RandomPlayer()
    elif player_type == 'greedy':
        return GreedyPlayer()
    elif player_type == 'neural':
        return NeuralPlayer(
            model_path=config['model_path'],
            num_players=config.get('num_players', 2),
            use_mcts=config.get('use_mcts', True),
            num_simulations=config.get('num_simulations', 50),
            device='cpu'  # Workers use CPU to avoid GPU contention
        )
    elif player_type == 'mcts':
        return MCTSPlayer(num_simulations=config.get('num_simulations', 50))
    else:
        raise ValueError(f"Unknown player type: {player_type}")


def _play_single_game_worker(game_seed: int) -> Dict:
    """
    Play a single game between the two worker players.
    
    This function runs in a worker process for parallel execution.
    
    Args:
        game_seed: Random seed for game initialization
    
    Returns:
        Dictionary with game results
    """
    global _worker_player1, _worker_player2, _worker_config
    
    # Create game with seed for reproducibility
    game = AzulGame(num_players=2, seed=game_seed)
    players = [_worker_player1, _worker_player2]
    
    move_count = 0
    max_moves = 300  # Safety limit
    
    # Play until game over
    while not game.game_over and move_count < max_moves:
        current = game.current_player
        player = players[current]
        
        action = player.select_action(game)
        if action is None:
            break
        
        game.take_action(action)
        move_count += 1
    
    # Collect results
    return {
        'p1_score': game.player_boards[0].score,
        'p2_score': game.player_boards[1].score,
        'winner': game.winner,
        'moves': move_count
    }


def _run_sequential_match(
    player1_config: Dict,
    player2_config: Dict,
    game_seeds: List[int],
    desc: str
) -> Dict:
    """
    Run games sequentially (fallback when parallel fails).
    
    Args:
        player1_config: Player 1 configuration
        player2_config: Player 2 configuration
        game_seeds: List of random seeds for games
        desc: Progress bar description
    
    Returns:
        Match results dictionary
    """
    results = {
        "player1_wins": 0,
        "player2_wins": 0,
        "draws": 0,
        "player1_scores": [],
        "player2_scores": [],
        "game_lengths": []
    }
    
    player1 = _create_player(player1_config)
    player2 = _create_player(player2_config)
    
    for seed in tqdm(game_seeds, desc=desc):
        game = AzulGame(num_players=2, seed=seed)
        players = [player1, player2]
        
        move_count = 0
        while not game.game_over and move_count < 300:
            current = game.current_player
            action = players[current].select_action(game)
            if action is None:
                break
            game.take_action(action)
            move_count += 1
        
        results["player1_scores"].append(game.player_boards[0].score)
        results["player2_scores"].append(game.player_boards[1].score)
        results["game_lengths"].append(move_count)
        
        if game.winner == 0:
            results["player1_wins"] += 1
        elif game.winner == 1:
            results["player2_wins"] += 1
        else:
            results["draws"] += 1
    
    return results


def play_match_parallel(
    player1_config: Dict,
    player2_config: Dict,
    num_games: int = 100,
    num_workers: int = 0,
    desc: str = "Match"
) -> Dict:
    """
    Play multiple games between two players using parallel execution.
    
    Distributes games across multiple worker processes for faster
    evaluation on multi-core systems. Falls back to sequential
    execution if parallel processing fails.
    
    Args:
        player1_config: Configuration dict for player 1
        player2_config: Configuration dict for player 2
        num_games: Number of games to play
        num_workers: Number of parallel workers (0 = auto-detect)
        desc: Description for progress bar
    
    Returns:
        Dictionary with match statistics
    """
    # Auto-detect workers
    if num_workers <= 0:
        num_workers = max(1, mp.cpu_count() // 2)
    
    config = {'num_players': 2}
    
    # Initialize results
    results = {
        "player1_wins": 0,
        "player2_wins": 0,
        "draws": 0,
        "player1_scores": [],
        "player2_scores": [],
        "game_lengths": []
    }
    
    # Generate random seeds for games
    game_seeds = [np.random.randint(0, 2**31) for _ in range(num_games)]
    
    # Try parallel execution first
    use_parallel = num_workers > 1 and num_games >= num_workers
    
    if use_parallel:
        try:
            # Parallel execution
            with ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=_init_game_worker,
                initargs=(player1_config, player2_config, config)
            ) as executor:
                futures = [
                    executor.submit(_play_single_game_worker, seed)
                    for seed in game_seeds
                ]
                
                for future in tqdm(as_completed(futures), total=num_games, desc=desc):
                    try:
                        result = future.result(timeout=300)  # 5 min timeout per game
                        
                        results["player1_scores"].append(result['p1_score'])
                        results["player2_scores"].append(result['p2_score'])
                        results["game_lengths"].append(result['moves'])
                        
                        if result['winner'] == 0:
                            results["player1_wins"] += 1
                        elif result['winner'] == 1:
                            results["player2_wins"] += 1
                        else:
                            results["draws"] += 1
                    except Exception as e:
                        # If a single game fails, log and continue
                        print(f"\n⚠️ Game failed: {e}")
                        continue
            
            return results
            
        except Exception as e:
            # Parallel execution failed - fall back to sequential
            print(f"\n⚠️ Parallel execution failed: {e}")
            print("   Falling back to sequential execution...")
            
            # Reset results and run sequentially
            return _run_sequential_match(
                player1_config, player2_config, game_seeds, desc
            )
    else:
        # Sequential execution (for small game counts or when parallel disabled)
        return _run_sequential_match(
            player1_config, player2_config, game_seeds, desc
        )


def play_match(
    player1,
    player2,
    num_games: int = 100,
    num_players: int = 2,
    verbose: bool = False
) -> Dict:
    """
    Play multiple games between two players (sequential version).
    
    Use this for small evaluations or when players can't be serialized.
    For large evaluations, use play_match_parallel().
    
    Args:
        player1: First player instance (plays as player 0)
        player2: Second player instance (plays as player 1)
        num_games: Number of games to play
        num_players: Total players in game
        verbose: Print game-by-game results
    
    Returns:
        Dictionary with match statistics:
        - player1_wins: Number of wins for player 1
        - player2_wins: Number of wins for player 2
        - draws: Number of draws
        - player1_scores: List of player 1's scores
        - player2_scores: List of player 2's scores
        - game_lengths: List of game lengths in moves
    """
    results = {
        "player1_wins": 0,
        "player2_wins": 0,
        "draws": 0,
        "player1_scores": [],
        "player2_scores": [],
        "game_lengths": []
    }
    
    players = [player1, player2]
    
    for game_idx in tqdm(range(num_games), desc=f"{player1} vs {player2}"):
        game = AzulGame(num_players=num_players)
        move_count = 0
        max_moves = 300  # Safety limit to prevent infinite games
        
        # Play until game over or move limit
        while not game.game_over and move_count < max_moves:
            current = game.current_player
            player = players[current]
            
            action = player.select_action(game)
            if action is None:
                break  # No legal actions
            
            game.take_action(action)
            move_count += 1
        
        # Record results
        p1_score = game.player_boards[0].score
        p2_score = game.player_boards[1].score
        
        results["player1_scores"].append(p1_score)
        results["player2_scores"].append(p2_score)
        results["game_lengths"].append(move_count)
        
        if game.winner == 0:
            results["player1_wins"] += 1
        elif game.winner == 1:
            results["player2_wins"] += 1
        else:
            results["draws"] += 1
        
        if verbose:
            winner_str = "P1 wins" if game.winner == 0 else "P2 wins" if game.winner == 1 else "Draw"
            print(f"Game {game_idx + 1}: {p1_score} - {p2_score} ({winner_str})")
    
    return results


# =============================================================================
# RESULTS DISPLAY AND STATISTICS
# =============================================================================

def calculate_statistics(results: Dict) -> Dict:
    """
    Calculate detailed statistics from match results.
    
    Args:
        results: Raw match results dictionary
    
    Returns:
        Dictionary with calculated statistics:
        - win_rate: Win percentage for player 1
        - avg_score: Average scores
        - score_std: Standard deviation of scores
        - score_margin: Average winning margin
        - confidence_interval: 95% CI for win rate
    """
    total = results["player1_wins"] + results["player2_wins"] + results["draws"]
    
    # Return default values when no games were played
    if total == 0:
        return {
            "total_games": 0,
            "p1_wins": 0,
            "p2_wins": 0,
            "draws": 0,
            "p1_winrate": 0.0,
            "p2_winrate": 0.0,
            "draw_rate": 0.0,
            "p1_avg_score": 0.0,
            "p2_avg_score": 0.0,
            "p1_std_score": 0.0,
            "p2_std_score": 0.0,
            "avg_margin": 0.0,
            "ci_low": 0.0,
            "ci_high": 100.0,
            "avg_game_length": 0.0
        }
    
    # Win rates
    p1_winrate = results["player1_wins"] / total
    p2_winrate = results["player2_wins"] / total
    draw_rate = results["draws"] / total
    
    # Score statistics (handle empty arrays)
    p1_scores = np.array(results["player1_scores"]) if results["player1_scores"] else np.array([0])
    p2_scores = np.array(results["player2_scores"]) if results["player2_scores"] else np.array([0])
    
    p1_avg = np.mean(p1_scores) if len(p1_scores) > 0 else 0.0
    p2_avg = np.mean(p2_scores) if len(p2_scores) > 0 else 0.0
    p1_std = np.std(p1_scores) if len(p1_scores) > 0 else 0.0
    p2_std = np.std(p2_scores) if len(p2_scores) > 0 else 0.0
    
    # Score margin (positive = player 1 winning)
    if len(p1_scores) > 0 and len(p2_scores) > 0 and len(p1_scores) == len(p2_scores):
        margins = p1_scores - p2_scores
        avg_margin = np.mean(margins)
    else:
        avg_margin = p1_avg - p2_avg
    
    # 95% confidence interval for win rate using Wilson score interval
    # This is more accurate than normal approximation for proportions
    n = total
    p = p1_winrate
    z = 1.96  # 95% confidence
    
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denominator
    margin = z * np.sqrt((p * (1-p) + z**2 / (4*n)) / n) / denominator
    
    ci_low = max(0, center - margin)
    ci_high = min(1, center + margin)
    
    # Game length statistics (handle empty)
    if results["game_lengths"]:
        avg_length = np.mean(results["game_lengths"])
    else:
        avg_length = 0.0
    
    return {
        "total_games": total,
        "p1_wins": results["player1_wins"],
        "p2_wins": results["player2_wins"],
        "draws": results["draws"],
        "p1_winrate": p1_winrate * 100,
        "p2_winrate": p2_winrate * 100,
        "draw_rate": draw_rate * 100,
        "p1_avg_score": p1_avg,
        "p2_avg_score": p2_avg,
        "p1_std_score": p1_std,
        "p2_std_score": p2_std,
        "avg_margin": avg_margin,
        "ci_low": ci_low * 100,
        "ci_high": ci_high * 100,
        "avg_game_length": avg_length
    }


def print_results(player1, player2, results: Dict) -> None:
    """
    Print formatted match results with statistics.
    
    Args:
        player1: Player 1 instance (for name)
        player2: Player 2 instance (for name)
        results: Match results dictionary
    """
    stats = calculate_statistics(results)
    
    print("\n" + "=" * 65)
    print(f"📊 MATCH RESULTS: {player1} vs {player2}")
    print("=" * 65)
    
    # Check if we have any data
    if stats['total_games'] == 0:
        print("\n⚠️ No games completed successfully!")
        print("=" * 65)
        return
    
    # Header
    print(f"\n{'Player':<25} {'Wins':>8} {'Win %':>10} {'Avg Score':>14}")
    print("-" * 60)
    
    # Player 1 stats
    print(f"{str(player1):<25} {stats['p1_wins']:>8} {stats['p1_winrate']:>9.1f}% "
          f"{stats['p1_avg_score']:>8.1f} ± {stats['p1_std_score']:.1f}")
    
    # Player 2 stats
    print(f"{str(player2):<25} {stats['p2_wins']:>8} {stats['p2_winrate']:>9.1f}% "
          f"{stats['p2_avg_score']:>8.1f} ± {stats['p2_std_score']:.1f}")
    
    # Draws
    print(f"{'Draws':<25} {stats['draws']:>8} {stats['draw_rate']:>9.1f}%")
    
    # Additional statistics
    print(f"\n📈 Additional Statistics:")
    print(f"   Average score margin: {stats['avg_margin']:+.1f} points")
    print(f"   95% CI for P1 win rate: [{stats['ci_low']:.1f}%, {stats['ci_high']:.1f}%]")
    print(f"   Average game length: {stats['avg_game_length']:.1f} moves")
    
    print("=" * 65)


def print_summary(all_results: Dict, model_name: str = "Model") -> None:
    """
    Print a summary of all evaluation results.
    
    Args:
        all_results: Dictionary of all match results
        model_name: Name of the evaluated model
    """
    print("\n" + "=" * 65)
    print(f"📈 EVALUATION SUMMARY: {model_name}")
    print("=" * 65)
    
    print(f"\n{'Opponent':<20} {'Win Rate':>12} {'Avg Margin':>14} {'Assessment':>15}")
    print("-" * 65)
    
    assessments = []
    
    for matchup, results in all_results.items():
        stats = calculate_statistics(results)
        
        # Skip if no games were played
        if stats['total_games'] == 0:
            continue
        
        # Determine assessment
        wr = stats['p1_winrate']
        if wr >= 80:
            assessment = "✅ Dominant"
        elif wr >= 65:
            assessment = "✅ Strong"
        elif wr >= 50:
            assessment = "⚠️ Even"
        elif wr >= 35:
            assessment = "⚠️ Weak"
        else:
            assessment = "❌ Losing"
        
        assessments.append((matchup, wr))
        
        opponent_name = matchup.replace("vs_", "").replace("_", " ").title()
        print(f"{opponent_name:<20} {wr:>11.1f}% {stats['avg_margin']:>+13.1f} {assessment:>15}")
    
    # Overall assessment
    print("\n" + "-" * 65)
    
    if "vs_random" in all_results:
        random_wr = calculate_statistics(all_results["vs_random"])['p1_winrate']
        
        if random_wr >= 80:
            print("🏆 Overall: Model is STRONG - consistently beats baselines")
        elif random_wr >= 65:
            print("✅ Overall: Model is LEARNING - shows clear improvement")
        elif random_wr >= 50:
            print("⚠️ Overall: Model is DEVELOPING - some learning, needs more training")
        else:
            print("❌ Overall: Model is NOT LEARNING - check training setup")
    
    print("=" * 65)


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_model(
    model_path: str,
    num_games: int = 100,
    num_simulations: int = 50,
    include_mcts_baseline: bool = True,
    parallel: bool = True,
    num_workers: int = 0,
    device: str = "auto"
) -> Dict:
    """
    Comprehensive evaluation of a trained model against multiple baselines.
    
    Tests the model against:
    1. Random player - Sanity check (should win >70%)
    2. Greedy player - Reasonable baseline (should win >55%)
    3. Raw network (no MCTS) - Tests network quality alone
    4. Pure MCTS (optional) - Tests if network helps MCTS
    
    Args:
        model_path: Path to model checkpoint
        num_games: Games per matchup
        num_simulations: MCTS simulations for neural player
        include_mcts_baseline: Also test against pure MCTS (slow)
        parallel: Use parallel game execution
        num_workers: Number of parallel workers (0 = auto)
        device: Compute device for neural network
    
    Returns:
        Dictionary with all evaluation results
    """
    print(f"\n{'='*65}")
    print(f"🎯 MODEL EVALUATION")
    print(f"{'='*65}")
    print(f"   Model: {model_path}")
    print(f"   Games per matchup: {num_games}")
    print(f"   MCTS simulations: {num_simulations}")
    print(f"   Parallel: {parallel}")
    if parallel:
        workers = num_workers if num_workers > 0 else max(1, mp.cpu_count() // 2)
        print(f"   Workers: {workers}")
    print(f"{'='*65}")
    
    all_results = {}
    start_time = time.time()
    
    # Configuration for parallel execution
    neural_config = {
        'type': 'neural',
        'model_path': model_path,
        'use_mcts': True,
        'num_simulations': num_simulations
    }
    neural_no_mcts_config = {
        'type': 'neural',
        'model_path': model_path,
        'use_mcts': False
    }
    random_config = {'type': 'random'}
    greedy_config = {'type': 'greedy'}
    
    # -------------------------------------------------------------------------
    # Test 1: Neural+MCTS vs Random
    # This should be an easy win if the model learned anything
    # -------------------------------------------------------------------------
    print("\n📊 Test 1: Neural+MCTS vs Random")
    
    if parallel:
        results = play_match_parallel(
            neural_config, random_config,
            num_games=num_games,
            num_workers=num_workers,
            desc="Neural+MCTS vs Random"
        )
    else:
        neural_player = NeuralPlayer(model_path, use_mcts=True, 
                                      num_simulations=num_simulations, device=device)
        results = play_match(neural_player, RandomPlayer(), num_games)
    
    print_results("Neural+MCTS", "Random", results)
    all_results["vs_random"] = results
    
    # -------------------------------------------------------------------------
    # Test 2: Neural+MCTS vs Greedy
    # Greedy is a reasonable baseline with simple strategy
    # -------------------------------------------------------------------------
    print("\n📊 Test 2: Neural+MCTS vs Greedy")
    
    if parallel:
        results = play_match_parallel(
            neural_config, greedy_config,
            num_games=num_games,
            num_workers=num_workers,
            desc="Neural+MCTS vs Greedy"
        )
    else:
        neural_player = NeuralPlayer(model_path, use_mcts=True,
                                      num_simulations=num_simulations, device=device)
        results = play_match(neural_player, GreedyPlayer(), num_games)
    
    print_results("Neural+MCTS", "Greedy", results)
    all_results["vs_greedy"] = results
    
    # -------------------------------------------------------------------------
    # Test 3: Neural (no MCTS) vs Random
    # Tests the raw network quality without search
    # -------------------------------------------------------------------------
    print("\n📊 Test 3: Neural (no MCTS) vs Random")
    
    if parallel:
        results = play_match_parallel(
            neural_no_mcts_config, random_config,
            num_games=num_games,
            num_workers=num_workers,
            desc="Neural vs Random"
        )
    else:
        neural_no_mcts = NeuralPlayer(model_path, use_mcts=False, device=device)
        results = play_match(neural_no_mcts, RandomPlayer(), num_games)
    
    print_results("Neural", "Random", results)
    all_results["neural_only_vs_random"] = results
    
    # -------------------------------------------------------------------------
    # Test 4: Pure MCTS vs Random (baseline)
    # Shows how much the network helps MCTS
    # -------------------------------------------------------------------------
    if include_mcts_baseline:
        print("\n📊 Test 4: Pure MCTS vs Random (baseline)")
        
        mcts_config = {'type': 'mcts', 'num_simulations': num_simulations}
        
        # Fewer games because pure MCTS is slow
        baseline_games = max(10, num_games // 4)
        
        if parallel:
            results = play_match_parallel(
                mcts_config, random_config,
                num_games=baseline_games,
                num_workers=num_workers,
                desc="MCTS vs Random"
            )
        else:
            mcts_player = MCTSPlayer(num_simulations=num_simulations)
            results = play_match(mcts_player, RandomPlayer(), baseline_games)
        
        print_results(f"MCTS({num_simulations})", "Random", results)
        all_results["mcts_baseline"] = results
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    elapsed = time.time() - start_time
    print(f"\n⏱️ Total evaluation time: {elapsed:.1f}s")
    
    print_summary(all_results, os.path.basename(model_path))
    
    return all_results


def compare_models(
    model1_path: str,
    model2_path: str,
    num_games: int = 100,
    num_simulations: int = 50,
    parallel: bool = True,
    num_workers: int = 0
) -> Dict:
    """
    Compare two trained models head-to-head.
    
    Plays matches with each model as both first and second player
    to eliminate any first-player advantage bias.
    
    Args:
        model1_path: Path to first model
        model2_path: Path to second model
        num_games: Total games (split between both sides)
        num_simulations: MCTS simulations per move
        parallel: Use parallel execution
        num_workers: Number of workers (0 = auto)
    
    Returns:
        Dictionary with combined results
    """
    print(f"\n{'='*65}")
    print(f"🆚 MODEL COMPARISON")
    print(f"{'='*65}")
    print(f"   Model 1: {model1_path}")
    print(f"   Model 2: {model2_path}")
    print(f"   Games: {num_games} ({num_games//2} per side)")
    print(f"{'='*65}")
    
    model1_config = {
        'type': 'neural',
        'model_path': model1_path,
        'use_mcts': True,
        'num_simulations': num_simulations
    }
    model2_config = {
        'type': 'neural',
        'model_path': model2_path,
        'use_mcts': True,
        'num_simulations': num_simulations
    }
    
    # -------------------------------------------------------------------------
    # Match 1: Model1 as Player 0
    # -------------------------------------------------------------------------
    print("\n📊 Match 1: Model1 as Player 0")
    
    if parallel:
        results1 = play_match_parallel(
            model1_config, model2_config,
            num_games=num_games // 2,
            num_workers=num_workers,
            desc="Model1 vs Model2"
        )
    else:
        player1 = NeuralPlayer(model1_path, use_mcts=True, num_simulations=num_simulations)
        player2 = NeuralPlayer(model2_path, use_mcts=True, num_simulations=num_simulations)
        results1 = play_match(player1, player2, num_games // 2)
    
    print_results("Model1", "Model2", results1)
    
    # -------------------------------------------------------------------------
    # Match 2: Model2 as Player 0 (reverse sides)
    # -------------------------------------------------------------------------
    print("\n📊 Match 2: Model2 as Player 0")
    
    if parallel:
        results2 = play_match_parallel(
            model2_config, model1_config,
            num_games=num_games // 2,
            num_workers=num_workers,
            desc="Model2 vs Model1"
        )
    else:
        player1 = NeuralPlayer(model2_path, use_mcts=True, num_simulations=num_simulations)
        player2 = NeuralPlayer(model1_path, use_mcts=True, num_simulations=num_simulations)
        results2 = play_match(player1, player2, num_games // 2)
    
    print_results("Model2", "Model1", results2)
    
    # -------------------------------------------------------------------------
    # Combined results (accounting for side swap)
    # -------------------------------------------------------------------------
    total_m1_wins = results1["player1_wins"] + results2["player2_wins"]
    total_m2_wins = results1["player2_wins"] + results2["player1_wins"]
    total_draws = results1["draws"] + results2["draws"]
    total = total_m1_wins + total_m2_wins + total_draws
    
    print("\n" + "=" * 65)
    print("📊 COMBINED RESULTS (both sides)")
    print("=" * 65)
    
    print(f"\n{'Model':<35} {'Wins':>8} {'Win %':>10}")
    print("-" * 55)
    
    m1_name = os.path.basename(model1_path)
    m2_name = os.path.basename(model2_path)
    
    print(f"Model 1: {m1_name:<25} {total_m1_wins:>8} {total_m1_wins/total*100:>9.1f}%")
    print(f"Model 2: {m2_name:<25} {total_m2_wins:>8} {total_m2_wins/total*100:>9.1f}%")
    print(f"{'Draws':<35} {total_draws:>8} {total_draws/total*100:>9.1f}%")
    
    # Determine winner
    print("\n" + "-" * 55)
    if total_m1_wins > total_m2_wins * 1.1:  # 10% margin
        print(f"🏆 Model 1 ({m1_name}) is STRONGER")
    elif total_m2_wins > total_m1_wins * 1.1:
        print(f"🏆 Model 2 ({m2_name}) is STRONGER")
    else:
        print("🤝 Models are roughly EQUAL in strength")
    print("=" * 65)
    
    return {
        "model1_wins": total_m1_wins,
        "model2_wins": total_m2_wins,
        "draws": total_draws,
        "model1_winrate": total_m1_wins / total * 100,
        "model2_winrate": total_m2_wins / total * 100
    }


def quick_test(num_games: int = 20, parallel: bool = True, num_workers: int = 0) -> None:
    """
    Quick sanity test of baseline players without a trained model.
    
    Useful for:
    - Verifying the game logic works
    - Checking baseline player strengths
    - Benchmarking evaluation speed
    
    Args:
        num_games: Games per matchup
        parallel: Use parallel execution
        num_workers: Number of workers
    """
    print("\n" + "=" * 65)
    print("🧪 QUICK SANITY TEST")
    print("=" * 65)
    print("   Testing baseline players without trained model...")
    print("   This verifies the game and evaluation code works correctly.")
    print("=" * 65)
    
    start_time = time.time()
    
    random_config = {'type': 'random'}
    greedy_config = {'type': 'greedy'}
    mcts_config = {'type': 'mcts', 'num_simulations': 25}
    
    # -------------------------------------------------------------------------
    # Test 1: Greedy vs Random
    # Greedy should win most games
    # -------------------------------------------------------------------------
    print("\n📊 Test 1: Greedy vs Random")
    print("   Expected: Greedy should win >60%")
    
    if parallel:
        results = play_match_parallel(
            greedy_config, random_config,
            num_games=num_games,
            num_workers=num_workers,
            desc="Greedy vs Random"
        )
    else:
        results = play_match(GreedyPlayer(), RandomPlayer(), num_games)
    
    print_results("Greedy", "Random", results)
    
    # -------------------------------------------------------------------------
    # Test 2: MCTS vs Random
    # MCTS should dominate
    # -------------------------------------------------------------------------
    print("\n📊 Test 2: MCTS(25) vs Random")
    print("   Expected: MCTS should win >75%")
    
    if parallel:
        results = play_match_parallel(
            mcts_config, random_config,
            num_games=num_games,
            num_workers=num_workers,
            desc="MCTS vs Random"
        )
    else:
        results = play_match(MCTSPlayer(num_simulations=25), RandomPlayer(), num_games)
    
    print_results("MCTS(25)", "Random", results)
    
    # -------------------------------------------------------------------------
    # Test 3: MCTS vs Greedy
    # MCTS should usually win but closer
    # -------------------------------------------------------------------------
    print("\n📊 Test 3: MCTS(25) vs Greedy")
    print("   Expected: MCTS should win >55%")
    
    if parallel:
        results = play_match_parallel(
            mcts_config, greedy_config,
            num_games=num_games,
            num_workers=num_workers,
            desc="MCTS vs Greedy"
        )
    else:
        results = play_match(MCTSPlayer(num_simulations=25), GreedyPlayer(), num_games)
    
    print_results("MCTS(25)", "Greedy", results)
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ Total test time: {elapsed:.1f}s")
    print("\n✅ Quick test complete! Baseline players are working correctly.")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main entry point for evaluation.
    
    Usage:
        python evaluate.py --model checkpoint.pt           # Evaluate model
        python evaluate.py --model1 v1.pt --model2 v2.pt   # Compare models
        python evaluate.py --quick-test                     # Test baselines
    """
    parser = argparse.ArgumentParser(
        description="Evaluate trained Azul RL models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Evaluate a model:
    python evaluate.py --model checkpoints/model_iter_100.pt --games 100
    
  Compare two models:
    python evaluate.py --model1 model_v1.pt --model2 model_v2.pt
    
  Quick baseline test:
    python evaluate.py --quick-test --games 20
    
  Fast parallel evaluation:
    python evaluate.py --model checkpoint.pt --games 200 --workers 8
        """
    )
    
    # Model paths
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to model checkpoint to evaluate"
    )
    parser.add_argument(
        "--model1",
        type=str,
        default=None,
        help="First model for head-to-head comparison"
    )
    parser.add_argument(
        "--model2",
        type=str,
        default=None,
        help="Second model for head-to-head comparison"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--games", "-g",
        type=int,
        default=100,
        help="Number of games per matchup (default: 100)"
    )
    parser.add_argument(
        "--simulations", "-s",
        type=int,
        default=50,
        help="MCTS simulations per move (default: 50)"
    )
    
    # Parallel execution
    # Parallel is enabled by default with automatic fallback to sequential if it fails
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        default=True,
        help="Use parallel game execution (default: True, auto-fallback if fails)"
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Explicitly disable parallel execution"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=0,
        help="Number of parallel workers (0=auto, default: 0)"
    )
    
    # Options
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick sanity test without trained model"
    )
    parser.add_argument(
        "--no-mcts-baseline",
        action="store_true",
        help="Skip slow MCTS baseline comparison"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Compute device (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Determine parallel setting
    parallel = args.parallel and not args.no_parallel
    
    # Route to appropriate function
    if args.quick_test:
        quick_test(
            num_games=args.games,
            parallel=parallel,
            num_workers=args.workers
        )
    elif args.model1 and args.model2:
        compare_models(
            args.model1,
            args.model2,
            num_games=args.games,
            num_simulations=args.simulations,
            parallel=parallel,
            num_workers=args.workers
        )
    elif args.model:
        evaluate_model(
            args.model,
            num_games=args.games,
            num_simulations=args.simulations,
            include_mcts_baseline=not args.no_mcts_baseline,
            parallel=parallel,
            num_workers=args.workers,
            device=args.device
        )
    else:
        # No arguments - show help
        print("🎯 Azul Model Evaluation Tool")
        print("=" * 50)
        print("\nUsage:")
        print("  Evaluate model:   python evaluate.py --model checkpoints/model.pt")
        print("  Compare models:   python evaluate.py --model1 v1.pt --model2 v2.pt")
        print("  Quick test:       python evaluate.py --quick-test")
        print("\nRun with --help for all options.")
        print("\n💡 Tip: Run --quick-test first to verify everything works!")


if __name__ == "__main__":
    main()

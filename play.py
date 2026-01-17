"""
Interactive play script for Azul.

Allows human vs AI, AI vs AI, or human vs human gameplay.
"""

import argparse
from typing import Optional, Tuple

from azul.game import AzulGame
from azul.env import AzulEnv
from azul.constants import TileColor, NUM_TILE_COLORS, PATTERN_LINES
from mcts.mcts import MCTSPlayer


def print_game_state(game: AzulGame) -> None:
    """Print the current game state in a readable format."""
    color_names = {
        TileColor.BLUE: 'B (Blue)',
        TileColor.YELLOW: 'Y (Yellow)', 
        TileColor.RED: 'R (Red)',
        TileColor.BLACK: 'K (Black)',
        TileColor.WHITE: 'W (White)',
    }
    color_chars = ['B', 'Y', 'R', 'K', 'W']
    
    print("\n" + "=" * 60)
    print(f"AZUL - Round {game.round_number}")
    print(f"Current Player: {game.current_player}")
    print("=" * 60)
    
    # Factories
    print("\n📦 FACTORIES:")
    for i, factory in enumerate(game.factories):
        if factory:
            tiles = ' '.join(color_chars[t] for t in factory)
            print(f"  [{i}] {tiles}")
        else:
            print(f"  [{i}] (empty)")
    
    # Center
    print("\n🎯 CENTER:")
    if game.center:
        tiles = ' '.join(color_chars[t] for t in game.center if t < 5)
        first = " + 1st Player Marker" if game.center_has_first_player else ""
        print(f"  {tiles}{first}")
    elif game.center_has_first_player:
        print("  1st Player Marker only")
    else:
        print("  (empty)")
    
    # Player boards
    print("\n👥 PLAYER BOARDS:")
    for player_idx, board in enumerate(game.player_boards):
        marker = ">>>" if player_idx == game.current_player else "   "
        print(f"\n{marker} Player {player_idx} (Score: {board.score})")
        
        # Pattern lines
        print("  Pattern Lines:")
        for row in range(PATTERN_LINES):
            count, color = board.pattern_lines[row]
            max_tiles = row + 1
            if color is not None:
                line = color_chars[color] * count + '.' * (max_tiles - count)
            else:
                line = '.' * max_tiles
            print(f"    Row {row}: [{line}]")
        
        # Wall
        print("  Wall:")
        wall_pattern = [
            ['B', 'Y', 'R', 'K', 'W'],
            ['W', 'B', 'Y', 'R', 'K'],
            ['K', 'W', 'B', 'Y', 'R'],
            ['R', 'K', 'W', 'B', 'Y'],
            ['Y', 'R', 'K', 'W', 'B'],
        ]
        for row in range(5):
            wall_row = ''
            for col in range(5):
                if board.wall[row, col]:
                    wall_row += wall_pattern[row][col]
                else:
                    wall_row += '·'
            print(f"    [{wall_row}]")
        
        # Floor
        floor_str = ''
        for tile in board.floor_line:
            if tile == TileColor.FIRST_PLAYER:
                floor_str += '1'
            else:
                floor_str += color_chars[tile]
        if not floor_str:
            floor_str = '(empty)'
        print(f"  Floor: [{floor_str}]")
    
    print()


def get_human_action(game: AzulGame) -> Tuple[int, TileColor, int]:
    """Get action input from human player."""
    legal_actions = game.get_legal_actions()
    
    print("\n📋 LEGAL ACTIONS:")
    for i, (source, color, dest) in enumerate(legal_actions):
        color_names = ['Blue', 'Yellow', 'Red', 'Black', 'White']
        source_name = f"Factory {source}" if source >= 0 else "Center"
        dest_name = f"Row {dest}" if dest >= 0 else "Floor"
        print(f"  [{i}] Take {color_names[color]} from {source_name} → {dest_name}")
    
    while True:
        try:
            choice = input("\nEnter action number: ").strip()
            idx = int(choice)
            if 0 <= idx < len(legal_actions):
                return legal_actions[idx]
            print("Invalid choice. Try again.")
        except ValueError:
            print("Please enter a number.")
        except KeyboardInterrupt:
            print("\nGame aborted.")
            exit(0)


def play_game(
    num_players: int = 2,
    human_players: list = None,
    mcts_simulations: int = 100,
    model_path: Optional[str] = None
) -> None:
    """
    Play a game of Azul.
    
    Args:
        num_players: Number of players (2-4)
        human_players: List of player indices that are human-controlled
        mcts_simulations: Number of MCTS simulations for AI
        model_path: Optional path to trained model
    """
    if human_players is None:
        human_players = [0]  # Default: player 0 is human
    
    game = AzulGame(num_players=num_players)
    
    # Create AI players
    ai_player = MCTSPlayer(
        num_simulations=mcts_simulations,
        temperature=0.1  # Slight randomness for variety
    )
    
    # Load model if provided
    if model_path:
        try:
            import torch
            from train import AzulNet
            
            network = AzulNet(num_players=num_players)
            checkpoint = torch.load(model_path)
            network.load_state_dict(checkpoint["network_state_dict"])
            network.eval()
            
            def policy_fn(state):
                probs, _ = network.predict(state, state.current_player)
                return probs
            
            def value_fn(state):
                _, value = network.predict(state, state.current_player)
                return value
            
            ai_player = MCTSPlayer(
                num_simulations=mcts_simulations,
                policy_fn=policy_fn,
                value_fn=value_fn
            )
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Using MCTS without neural network.")
    
    print("\n🎮 AZUL GAME START 🎮")
    print(f"Players: {num_players}")
    print(f"Human players: {human_players}")
    print("-" * 40)
    
    while not game.game_over:
        print_game_state(game)
        
        current = game.current_player
        
        if current in human_players:
            # Human turn
            action = get_human_action(game)
        else:
            # AI turn
            print(f"🤖 AI Player {current} is thinking...")
            action = ai_player.select_action(game)
            
            color_names = ['Blue', 'Yellow', 'Red', 'Black', 'White']
            source, color, dest = action
            source_name = f"Factory {source}" if source >= 0 else "Center"
            dest_name = f"Row {dest}" if dest >= 0 else "Floor"
            print(f"   AI chooses: {color_names[color]} from {source_name} → {dest_name}")
        
        result = game.take_action(action)
        
        if result.get('round_ended'):
            print("\n🔄 ROUND ENDED - Scoring...\n")
    
    # Game over
    print("\n" + "=" * 60)
    print("🏆 GAME OVER! 🏆")
    print("=" * 60)
    
    print("\nFinal Scores:")
    for i, board in enumerate(game.player_boards):
        winner_mark = " 🥇" if i == game.winner else ""
        print(f"  Player {i}: {board.score} points{winner_mark}")
    
    if game.winner is not None:
        print(f"\n🎉 Player {game.winner} wins! 🎉")
    else:
        print("\n🤝 It's a tie!")


def watch_ai_game(
    num_players: int = 2,
    mcts_simulations: int = 100,
    delay: float = 1.0
) -> None:
    """Watch AI players play against each other."""
    import time
    
    game = AzulGame(num_players=num_players)
    ai_player = MCTSPlayer(num_simulations=mcts_simulations)
    
    print("\n🤖 AI VS AI GAME 🤖")
    print("-" * 40)
    
    move_count = 0
    while not game.game_over:
        print_game_state(game)
        
        current = game.current_player
        print(f"🤖 AI Player {current} is thinking...")
        
        action = ai_player.select_action(game)
        
        color_names = ['Blue', 'Yellow', 'Red', 'Black', 'White']
        source, color, dest = action
        source_name = f"Factory {source}" if source >= 0 else "Center"
        dest_name = f"Row {dest}" if dest >= 0 else "Floor"
        print(f"   Move {move_count + 1}: {color_names[color]} from {source_name} → {dest_name}")
        
        game.take_action(action)
        move_count += 1
        
        time.sleep(delay)
    
    print("\n🏆 GAME OVER! 🏆")
    print("\nFinal Scores:")
    for i, board in enumerate(game.player_boards):
        winner_mark = " 🥇" if i == game.winner else ""
        print(f"  Player {i}: {board.score} points{winner_mark}")


def main():
    parser = argparse.ArgumentParser(description="Play Azul")
    parser.add_argument(
        "--mode",
        choices=["human", "watch", "pvp"],
        default="human",
        help="Game mode: human (vs AI), watch (AI vs AI), pvp (human vs human)"
    )
    parser.add_argument(
        "--players",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of players"
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=100,
        help="MCTS simulations for AI"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between AI moves in watch mode"
    )
    
    args = parser.parse_args()
    
    if args.mode == "human":
        play_game(
            num_players=args.players,
            human_players=[0],
            mcts_simulations=args.simulations,
            model_path=args.model
        )
    elif args.mode == "watch":
        watch_ai_game(
            num_players=args.players,
            mcts_simulations=args.simulations,
            delay=args.delay
        )
    elif args.mode == "pvp":
        play_game(
            num_players=args.players,
            human_players=list(range(args.players)),
            mcts_simulations=args.simulations
        )


if __name__ == "__main__":
    main()

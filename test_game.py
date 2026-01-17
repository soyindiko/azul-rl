"""
Basic tests for Azul game implementation.

Run with: python test_game.py
"""

import numpy as np
from azul.game import AzulGame, PlayerBoard
from azul.env import AzulEnv
from azul.constants import TileColor, PATTERN_LINES
from mcts.mcts import MCTS, MCTSPlayer


def test_game_initialization():
    """Test basic game initialization."""
    print("Testing game initialization...")
    
    game = AzulGame(num_players=2)
    
    assert game.num_players == 2
    assert game.num_factories == 5
    assert len(game.factories) == 5
    assert game.current_player == 0
    assert not game.game_over
    
    # Check factories are filled
    total_tiles = sum(len(f) for f in game.factories)
    assert total_tiles == 20  # 5 factories * 4 tiles
    
    print("✓ Game initialization passed")


def test_legal_actions():
    """Test legal action generation."""
    print("Testing legal actions...")
    
    game = AzulGame(num_players=2)
    actions = game.get_legal_actions()
    
    assert len(actions) > 0
    
    # All actions should be valid (source, color, dest) tuples
    for source, color, dest in actions:
        assert -1 <= source < game.num_factories
        assert 0 <= color < 5
        assert -1 <= dest < PATTERN_LINES
    
    print(f"✓ Legal actions passed ({len(actions)} actions available)")


def test_take_action():
    """Test taking an action."""
    print("Testing take action...")
    
    game = AzulGame(num_players=2, seed=42)
    initial_player = game.current_player
    
    actions = game.get_legal_actions()
    action = actions[0]
    
    result = game.take_action(action)
    
    assert 'player' in result
    assert 'tiles_taken' in result
    assert result['tiles_taken'] > 0
    
    print(f"✓ Take action passed (took {result['tiles_taken']} tiles)")


def test_round_completion():
    """Test playing through a complete round."""
    print("Testing round completion...")
    
    game = AzulGame(num_players=2, seed=42)
    
    move_count = 0
    while True:
        actions = game.get_legal_actions()
        if not actions:
            break
        
        action = actions[0]
        result = game.take_action(action)
        move_count += 1
        
        if result.get('round_ended'):
            break
        
        if move_count > 100:
            raise Exception("Round didn't complete in reasonable time")
    
    print(f"✓ Round completion passed ({move_count} moves)")


def test_full_game():
    """Test playing a complete game."""
    print("Testing full game...")
    
    game = AzulGame(num_players=2, seed=42)
    
    move_count = 0
    while not game.game_over:
        actions = game.get_legal_actions()
        if not actions:
            break
        
        # Take first legal action
        action = actions[0]
        game.take_action(action)
        move_count += 1
        
        if move_count > 500:
            raise Exception("Game didn't complete in reasonable time")
    
    assert game.game_over
    
    final_scores = [pb.score for pb in game.player_boards]
    print(f"✓ Full game passed ({move_count} moves)")
    print(f"  Final scores: {final_scores}")
    print(f"  Winner: Player {game.winner}")


def test_env_creation():
    """Test PettingZoo environment creation."""
    print("Testing environment creation...")
    
    env = AzulEnv(num_players=2)
    env.reset()
    
    assert len(env.agents) == 2
    assert env.agent_selection is not None
    
    # Check observation
    obs = env.observe(env.agent_selection)
    assert "factories" in obs
    assert "center" in obs
    assert "my_wall" in obs
    assert "action_mask" in obs
    
    print("✓ Environment creation passed")


def test_env_gameplay():
    """Test playing through environment."""
    print("Testing environment gameplay...")
    
    env = AzulEnv(num_players=2)
    env.reset(seed=42)
    
    move_count = 0
    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        
        if term or trunc:
            action = None
        else:
            # Get valid actions from mask
            action_mask = obs["action_mask"]
            valid_actions = [i for i, v in enumerate(action_mask) if v]
            action = valid_actions[0] if valid_actions else 0
        
        env.step(action)
        move_count += 1
        
        if move_count > 500:
            break
    
    env.close()
    print(f"✓ Environment gameplay passed ({move_count} steps)")


def test_mcts():
    """Test MCTS search."""
    print("Testing MCTS...")
    
    game = AzulGame(num_players=2, seed=42)
    
    mcts = MCTS(num_simulations=10)
    action = mcts.get_action(game)
    
    # Check action is legal
    legal_actions = game.get_legal_actions()
    assert action in legal_actions
    
    print(f"✓ MCTS passed (selected action: {action})")


def test_mcts_player():
    """Test MCTS player in a short game."""
    print("Testing MCTS player...")
    
    game = AzulGame(num_players=2, seed=42)
    player = MCTSPlayer(num_simulations=10)
    
    # Play a few moves
    for _ in range(5):
        if game.game_over:
            break
        
        action = player.select_action(game)
        legal_actions = game.get_legal_actions()
        assert action in legal_actions
        
        game.take_action(action)
    
    print("✓ MCTS player passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 50)
    print("AZUL GAME TESTS")
    print("=" * 50 + "\n")
    
    tests = [
        test_game_initialization,
        test_legal_actions,
        test_take_action,
        test_round_completion,
        test_full_game,
        test_env_creation,
        test_env_gameplay,
        test_mcts,
        test_mcts_player,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

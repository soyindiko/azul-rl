# 🎨 Azul - Reinforcement Learning Environment

A faithful implementation of the [Azul](https://boardgamegeek.com/boardgame/230802/azul) board game as a multi-agent reinforcement learning environment using [PettingZoo](https://pettingzoo.farama.org/), with Monte Carlo Tree Search (MCTS) for training and gameplay.

## 🎯 Features

- **Complete Azul Implementation**: Full game rules including pattern lines, wall placement, scoring, and end-game bonuses
- **PettingZoo Environment**: Standard AEC (Alternating Environment with Communication) interface for RL research
- **MCTS Algorithm**: Monte Carlo Tree Search with UCB exploration for intelligent gameplay
- **AlphaZero-style Training**: Neural network integration for policy and value prediction
- **Interactive Play**: Command-line interface for human vs AI, AI vs AI, or human vs human games

## 📁 Project Structure

```
azul/
├── azul/                    # Core game module
│   ├── __init__.py
│   ├── constants.py         # Game constants and rules
│   ├── game.py              # Core game logic
│   └── env.py               # PettingZoo environment
├── mcts/                    # MCTS implementation
│   ├── __init__.py
│   ├── node.py              # MCTS tree node
│   └── mcts.py              # MCTS algorithm
├── train.py                 # Training script
├── play.py                  # Interactive play script
├── requirements.txt         # Dependencies
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
cd azul

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Play Against AI

```bash
# Human vs AI (default)
python play.py

# With more MCTS simulations (stronger AI)
python play.py --simulations 500

# Watch AI vs AI
python play.py --mode watch

# Human vs Human
python play.py --mode pvp

# 4 player game
python play.py --players 4
```

### Train an Agent

```bash
# Basic training
python train.py

# Custom training parameters
python train.py \
    --num-iterations 100 \
    --games-per-iteration 20 \
    --num-simulations 100 \
    --device cuda  # Use GPU if available

# Play with trained model
python play.py --model checkpoints/model_iter_100.pt
```

## 🎮 Game Rules (Azul)

### Overview
Azul is an abstract strategy game where players take turns drafting colored tiles from factories to complete patterns on their board.

### Turn Structure
1. **Factory Offer**: Take all tiles of ONE color from:
   - A factory display (remaining tiles go to center)
   - The center of the table
2. **Pattern Line Placement**: Place tiles on one of your 5 pattern lines (or floor)
3. First player to take from center gets the first player marker (penalty tile)

### Scoring
- **During Round**: Completed pattern lines move one tile to the wall
  - Score points based on adjacent tiles
- **Floor Penalties**: -1, -1, -2, -2, -2, -3, -3 for tiles 1-7
- **End Game Bonuses**:
  - Complete horizontal row: +2 points
  - Complete vertical column: +7 points
  - All 5 of one color: +10 points

### Game End
The game ends when any player completes a horizontal row on their wall.

## 🔧 Using the Environment

```python
from azul.env import AzulEnv

# Create environment
env = AzulEnv(num_players=2, render_mode="human")
env.reset()

# Game loop
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    
    if termination or truncation:
        action = None
    else:
        # Get valid actions from observation
        action_mask = observation["action_mask"]
        valid_actions = [i for i, valid in enumerate(action_mask) if valid]
        action = random.choice(valid_actions)
    
    env.step(action)

env.close()
```

## 🧠 MCTS Integration

```python
from azul.game import AzulGame
from mcts.mcts import MCTSPlayer

# Create game and player
game = AzulGame(num_players=2)
player = MCTSPlayer(num_simulations=100)

# Get best action
action = player.select_action(game)
game.take_action(action)
```

## 📊 Observation Space

| Component | Shape | Description |
|-----------|-------|-------------|
| `factories` | (N, 5) | Tile counts per factory |
| `center` | (5,) | Tile counts in center |
| `center_has_first` | (1,) | First player marker present |
| `my_pattern_lines` | (5, 2) | Count and color per line |
| `my_wall` | (5, 5) | Wall tile placement |
| `my_floor` | (7,) | Floor line tiles |
| `my_score` | (1,) | Current score |
| `opponent_walls` | (P-1, 5, 5) | Opponent wall states |
| `opponent_scores` | (P-1,) | Opponent scores |
| `action_mask` | (A,) | Valid action mask |

## 🏗️ Architecture

### Neural Network (AlphaZero-style)
- **Encoder**: 4-layer MLP with batch normalization
- **Policy Head**: Action probability distribution
- **Value Head**: State value estimation [-1, 1]

### MCTS
- UCB formula with PUCT modification for neural network priors
- Configurable exploration constant (c_puct)
- Support for random rollouts or neural network evaluation

## 📈 Future Improvements

- [ ] Convolutional encoder for spatial features
- [ ] Distributed self-play
- [ ] Elo rating system for model comparison
- [ ] Web interface for gameplay
- [ ] Support for variant rules

## 📚 References

- [Azul Board Game](https://boardgamegeek.com/boardgame/230802/azul)
- [PettingZoo Documentation](https://pettingzoo.farama.org/)
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)
- [MCTS Survey](https://ieeexplore.ieee.org/document/6145622)

## 📄 License

MIT License - Feel free to use for research and personal projects.

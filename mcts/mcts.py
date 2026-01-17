"""
Monte Carlo Tree Search algorithm implementation.
"""

import numpy as np
from typing import Optional, Callable, Dict, Tuple, List, Any
from copy import deepcopy

from mcts.node import MCTSNode
from azul.game import AzulGame


class MCTS:
    """
    Monte Carlo Tree Search implementation for Azul.
    
    Can be used standalone with random rollouts or integrated with a
    neural network for policy and value estimation (AlphaZero-style).
    """
    
    def __init__(
        self,
        c_puct: float = 1.41,
        num_simulations: int = 100,
        policy_fn: Optional[Callable] = None,
        value_fn: Optional[Callable] = None,
        use_rollouts: bool = True,
        rollout_depth: int = 50
    ):
        """
        Initialize MCTS.
        
        Args:
            c_puct: Exploration constant for UCB
            num_simulations: Number of simulations per search
            policy_fn: Optional function (game_state) -> Dict[action, prior]
                      For neural network policy prediction
            value_fn: Optional function (game_state) -> float
                     For neural network value prediction
            use_rollouts: Whether to use random rollouts for value estimation
                         (only when value_fn is None)
            rollout_depth: Maximum depth for random rollouts
        """
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.policy_fn = policy_fn
        self.value_fn = value_fn
        self.use_rollouts = use_rollouts
        self.rollout_depth = rollout_depth
        
        self.root: Optional[MCTSNode] = None
    
    def search(self, game_state: AzulGame) -> MCTSNode:
        """
        Perform MCTS search from given game state.
        
        Args:
            game_state: Current game state to search from
        
        Returns:
            Root node after search
        """
        # Create root node
        self.root = MCTSNode(game_state=game_state.copy())
        
        # Get action priors for root
        action_priors = self._get_action_priors(self.root.game_state)
        self.root.expand(action_priors)
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = self.root
            
            # Selection: traverse tree to leaf
            # Continue only if node is expanded AND has children to select from
            while node.is_expanded and node.children and not node.is_terminal:
                node = node.select_child(self.c_puct)
            
            # Check if we're at a terminal state or no legal actions
            if node.is_terminal or not node.legal_actions:
                value = self._terminal_value(node)
            else:
                # Expansion: expand the leaf node if not already expanded
                if not node.is_expanded:
                    action_priors = self._get_action_priors(node.game_state)
                    node.expand(action_priors)
                
                # Evaluation: evaluate the current node
                value = self._evaluate(node)
            
            # Backpropagation
            player = node.game_state.current_player
            node.backpropagate(value, player)
        
        return self.root
    
    def get_action(
        self,
        game_state: AzulGame,
        temperature: float = 0.0
    ) -> Tuple:
        """
        Get best action for given game state.
        
        Args:
            game_state: Current game state
            temperature: Temperature for action selection
        
        Returns:
            Best action tuple
        """
        root = self.search(game_state)
        return root.best_action(temperature)
    
    def get_action_probs(
        self,
        game_state: AzulGame,
        temperature: float = 1.0
    ) -> Dict[Tuple, float]:
        """
        Get action probabilities for training.
        
        Args:
            game_state: Current game state
            temperature: Temperature for probability calculation
        
        Returns:
            Dictionary mapping actions to probabilities
        """
        root = self.search(game_state)
        return root.get_action_probs(temperature)
    
    def _get_action_priors(self, game_state: AzulGame) -> Dict[Tuple, float]:
        """Get action priors from policy network or uniform."""
        if self.policy_fn is not None:
            return self.policy_fn(game_state)
        
        # Uniform priors
        actions = game_state.get_legal_actions()
        if not actions:
            return {}
        return {a: 1.0 / len(actions) for a in actions}
    
    def _evaluate(self, node: MCTSNode) -> float:
        """
        Evaluate a leaf node.
        
        Uses value function if available, otherwise random rollout.
        """
        if self.value_fn is not None:
            return self.value_fn(node.game_state)
        
        if self.use_rollouts:
            return self._rollout(node.game_state)
        
        # Heuristic evaluation based on scores
        return self._heuristic_value(node.game_state)
    
    def _rollout(self, game_state: AzulGame) -> float:
        """
        Perform random rollout to estimate value.
        
        Returns value from perspective of current player.
        """
        game = game_state.copy()
        original_player = game.current_player
        
        for _ in range(self.rollout_depth):
            if game.game_over:
                break
            
            actions = game.get_legal_actions()
            if not actions:
                break
            
            # Random action selection
            action = actions[np.random.randint(len(actions))]
            game.take_action(action)
        
        return self._get_value(game, original_player)
    
    def _terminal_value(self, node: MCTSNode) -> float:
        """Get value of terminal node."""
        game = node.game_state
        current_player = game.current_player
        
        if game.winner is not None:
            if game.winner == current_player:
                return 1.0
            else:
                return -1.0
        
        # Game over but no clear winner (tie)
        return 0.0
    
    def _heuristic_value(self, game_state: AzulGame) -> float:
        """
        Calculate heuristic value based on current game state.
        
        Uses score difference and board position.
        """
        current_player = game_state.current_player
        my_board = game_state.player_boards[current_player]
        
        # Average opponent score
        opponent_scores = [
            pb.score for i, pb in enumerate(game_state.player_boards)
            if i != current_player
        ]
        avg_opponent_score = np.mean(opponent_scores) if opponent_scores else 0
        
        # Score advantage
        score_diff = my_board.score - avg_opponent_score
        
        # Normalize to [-1, 1] range (rough estimate)
        max_expected_diff = 50  # Rough estimate of max score difference
        value = np.tanh(score_diff / max_expected_diff)
        
        return value
    
    def _get_value(self, game_state: AzulGame, player: int) -> float:
        """Get value of game state from perspective of given player."""
        if game_state.game_over:
            if game_state.winner == player:
                return 1.0
            elif game_state.winner is not None:
                return -1.0
            return 0.0
        
        return self._heuristic_value_for_player(game_state, player)
    
    def _heuristic_value_for_player(
        self, game_state: AzulGame, player: int
    ) -> float:
        """Calculate heuristic value from perspective of specific player."""
        my_board = game_state.player_boards[player]
        
        opponent_scores = [
            pb.score for i, pb in enumerate(game_state.player_boards)
            if i != player
        ]
        avg_opponent_score = np.mean(opponent_scores) if opponent_scores else 0
        
        score_diff = my_board.score - avg_opponent_score
        max_expected_diff = 50
        
        return np.tanh(score_diff / max_expected_diff)
    
    def update_with_move(self, action: Tuple) -> None:
        """
        Update MCTS tree after a move is made.
        
        Reuses subtree if the action was explored.
        
        Args:
            action: Action that was taken
        """
        if self.root is None:
            return
        
        if action in self.root.children:
            # Reuse subtree
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            # Discard tree
            self.root = None


class MCTSPlayer:
    """
    A player that uses MCTS to select actions.
    
    Wrapper class for easy integration with the environment.
    """
    
    def __init__(
        self,
        num_simulations: int = 100,
        c_puct: float = 1.41,
        temperature: float = 0.0,
        policy_fn: Optional[Callable] = None,
        value_fn: Optional[Callable] = None
    ):
        """
        Initialize MCTS player.
        
        Args:
            num_simulations: Number of MCTS simulations per move
            c_puct: Exploration constant
            temperature: Temperature for action selection (0 = greedy)
            policy_fn: Optional neural network policy function
            value_fn: Optional neural network value function
        """
        self.mcts = MCTS(
            c_puct=c_puct,
            num_simulations=num_simulations,
            policy_fn=policy_fn,
            value_fn=value_fn
        )
        self.temperature = temperature
    
    def select_action(self, game_state: AzulGame) -> Tuple:
        """
        Select an action for the current game state.
        
        Args:
            game_state: Current game state
        
        Returns:
            Selected action tuple
        """
        return self.mcts.get_action(game_state, self.temperature)
    
    def get_action_probs(
        self, game_state: AzulGame, temperature: float = 1.0
    ) -> Dict[Tuple, float]:
        """
        Get action probabilities for training.
        
        Args:
            game_state: Current game state
            temperature: Temperature for probabilities
        
        Returns:
            Action probability distribution
        """
        return self.mcts.get_action_probs(game_state, temperature)
    
    def reset(self) -> None:
        """Reset the MCTS tree."""
        self.mcts.root = None

"""
MCTS Node implementation.
"""

import math
import numpy as np
from typing import Optional, List, Tuple, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from azul.game import AzulGame


class MCTSNode:
    """
    Node in the Monte Carlo Tree Search tree.
    
    Each node represents a game state after taking a specific action.
    """
    
    def __init__(
        self,
        game_state: 'AzulGame',
        parent: Optional['MCTSNode'] = None,
        action: Optional[Tuple[int, int, int]] = None,
        prior: float = 0.0
    ):
        """
        Initialize MCTS node.
        
        Args:
            game_state: Copy of game state at this node
            parent: Parent node (None for root)
            action: Action that led to this state from parent
            prior: Prior probability from policy network (for RL integration)
        """
        self.game_state = game_state
        self.parent = parent
        self.action = action
        self.prior = prior
        
        # Statistics
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.children: Dict[Tuple, 'MCTSNode'] = {}
        
        # Cache legal actions
        self._legal_actions: Optional[List[Tuple]] = None
        self._is_expanded: bool = False
    
    @property
    def legal_actions(self) -> List[Tuple]:
        """Get legal actions from this state (cached)."""
        if self._legal_actions is None:
            self._legal_actions = self.game_state.get_legal_actions()
        return self._legal_actions
    
    @property
    def is_terminal(self) -> bool:
        """Check if this node represents a terminal state."""
        return self.game_state.game_over
    
    @property
    def is_expanded(self) -> bool:
        """Check if node has been expanded."""
        return self._is_expanded
    
    @property
    def q_value(self) -> float:
        """Average value (Q) of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, c_puct: float = 1.41) -> float:
        """
        Calculate UCB (Upper Confidence Bound) score.
        
        Uses PUCT formula for neural network integration:
        Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        
        Args:
            c_puct: Exploration constant (higher = more exploration)
        
        Returns:
            UCB score
        """
        if self.parent is None:
            return 0.0
        
        # Exploitation term
        q = self.q_value
        
        # Exploration term
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        
        return q + exploration
    
    def expand(self, action_priors: Optional[Dict[Tuple, float]] = None) -> None:
        """
        Expand this node by creating children for all legal actions.
        
        Args:
            action_priors: Optional dict mapping actions to prior probabilities.
                          If None, uniform priors are used.
        """
        if self._is_expanded or self.is_terminal:
            return
        
        actions = self.legal_actions
        
        if not actions:
            self._is_expanded = True
            return
        
        # Default to uniform priors
        if action_priors is None:
            uniform_prior = 1.0 / len(actions)
            action_priors = {a: uniform_prior for a in actions}
        
        for action in actions:
            if action not in self.children:
                # Create child game state
                child_game = self.game_state.copy()
                child_game.take_action(action)
                
                prior = action_priors.get(action, 1.0 / len(actions))
                
                child_node = MCTSNode(
                    game_state=child_game,
                    parent=self,
                    action=action,
                    prior=prior
                )
                self.children[action] = child_node
        
        self._is_expanded = True
    
    def select_child(self, c_puct: float = 1.41) -> 'MCTSNode':
        """
        Select the child with highest UCB score.
        
        Args:
            c_puct: Exploration constant
        
        Returns:
            Child node with highest UCB score
        """
        if not self.children:
            raise ValueError("Cannot select child from unexpanded node")
        
        best_score = float('-inf')
        best_child = None
        
        for child in self.children.values():
            score = child.ucb_score(c_puct)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def backpropagate(self, value: float, player: int) -> None:
        """
        Backpropagate the result up the tree.
        
        Args:
            value: Value to propagate (from perspective of player who just moved)
            player: Player who achieved this value
        """
        self.visit_count += 1
        
        # Value is from perspective of the player who made the move
        # We need to flip it when propagating to opponent's turns
        current_player = self.game_state.current_player
        
        # If this node's move was made by the same player, use positive value
        # Otherwise, use negative (it's good for opponent = bad for us)
        if self.parent is not None:
            parent_player = self.parent.game_state.current_player
            if parent_player == player:
                self.value_sum += value
            else:
                self.value_sum -= value
        
        if self.parent is not None:
            self.parent.backpropagate(value, player)
    
    def best_action(self, temperature: float = 0.0) -> Tuple:
        """
        Select the best action based on visit counts.
        
        Args:
            temperature: Temperature for action selection.
                        0 = greedy (most visited)
                        > 0 = sample proportionally to visit counts^(1/temp)
        
        Returns:
            Best action tuple
        """
        if not self.children:
            raise ValueError("No children to select from")
        
        actions = list(self.children.keys())
        visit_counts = np.array([
            self.children[a].visit_count for a in actions
        ])
        
        if temperature == 0:
            # Greedy: select most visited
            best_idx = np.argmax(visit_counts)
        else:
            # Sample proportionally
            if visit_counts.sum() == 0:
                probs = np.ones(len(actions)) / len(actions)
            else:
                counts_temp = np.power(visit_counts.astype(float), 1.0 / temperature)
                probs = counts_temp / counts_temp.sum()
            best_idx = np.random.choice(len(actions), p=probs)
        
        return actions[best_idx]
    
    def get_action_probs(self, temperature: float = 1.0) -> Dict[Tuple, float]:
        """
        Get action probabilities based on visit counts.
        
        Args:
            temperature: Temperature for probability calculation
        
        Returns:
            Dictionary mapping actions to probabilities
        """
        if not self.children:
            return {}
        
        actions = list(self.children.keys())
        visit_counts = np.array([
            self.children[a].visit_count for a in actions
        ], dtype=float)
        
        if visit_counts.sum() == 0:
            probs = np.ones(len(actions)) / len(actions)
        elif temperature == 0:
            probs = np.zeros(len(actions))
            probs[np.argmax(visit_counts)] = 1.0
        else:
            counts_temp = np.power(visit_counts, 1.0 / temperature)
            probs = counts_temp / counts_temp.sum()
        
        return {a: p for a, p in zip(actions, probs)}
    
    def __repr__(self) -> str:
        return (
            f"MCTSNode(action={self.action}, "
            f"visits={self.visit_count}, "
            f"q={self.q_value:.3f}, "
            f"children={len(self.children)})"
        )

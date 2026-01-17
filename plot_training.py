#!/usr/bin/env python3
"""
Visualization script for Azul RL training history.

Plots training metrics over iterations:
- Loss curves (policy, value, total)
- Learning rate schedule
- Games played and buffer size
- Training speed

Usage:
    python plot_training.py                              # Default: checkpoints/training_history.json
    python plot_training.py --history path/to/history.json
    python plot_training.py --save training_plot.png    # Save to file instead of showing
"""

import argparse
import json
import os
import sys

def load_history(path: str) -> list:
    """Load training history from JSON file."""
    if not os.path.exists(path):
        print(f"❌ History file not found: {path}")
        sys.exit(1)
    
    with open(path, 'r') as f:
        history = json.load(f)
    
    if not history:
        print("❌ History file is empty!")
        sys.exit(1)
    
    return history


def plot_training(history: list, save_path: str = None, show: bool = True):
    """
    Create training visualization plots.
    
    Args:
        history: List of training history entries
        save_path: If provided, save plot to this path
        show: If True, display the plot interactively
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        if not show:
            matplotlib.use('Agg')  # Non-interactive backend for saving
    except ImportError:
        print("❌ matplotlib is required for plotting.")
        print("   Install it with: pip install matplotlib")
        sys.exit(1)
    
    # Extract data
    iterations = [h['iteration'] for h in history]
    policy_loss = [h['policy_loss'] for h in history]
    value_loss = [h['value_loss'] for h in history]
    total_loss = [h['total_loss'] for h in history]
    learning_rate = [h['learning_rate'] for h in history]
    games_played = [h['games_played'] for h in history]
    buffer_size = [h['buffer_size'] for h in history]
    iter_time = [h.get('iteration_time', 0) for h in history]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Azul RL Training Progress', fontsize=14, fontweight='bold')
    
    # Plot 1: Losses
    ax1 = axes[0, 0]
    ax1.plot(iterations, policy_loss, label='Policy Loss', color='#2E86AB', linewidth=1.5)
    ax1.plot(iterations, value_loss, label='Value Loss', color='#A23B72', linewidth=1.5)
    ax1.plot(iterations, total_loss, label='Total Loss', color='#F18F01', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Losses')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(min(iterations), max(iterations))
    
    # Plot 2: Learning Rate
    ax2 = axes[0, 1]
    ax2.plot(iterations, learning_rate, color='#28965A', linewidth=1.5)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule (Cosine Annealing)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(min(iterations), max(iterations))
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Plot 3: Games and Buffer
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(iterations, games_played, label='Games Played', color='#5C4D7D', linewidth=1.5)
    line2 = ax3_twin.plot(iterations, buffer_size, label='Buffer Size', color='#E84855', linewidth=1.5)
    
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Games Played', color='#5C4D7D')
    ax3_twin.set_ylabel('Buffer Size', color='#E84855')
    ax3.set_title('Training Data Accumulation')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(min(iterations), max(iterations))
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')
    
    # Plot 4: Training Speed
    ax4 = axes[1, 1]
    if any(t > 0 for t in iter_time):
        ax4.plot(iterations, iter_time, color='#087E8B', linewidth=1.5)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Time (seconds)')
        ax4.set_title('Iteration Time')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(min(iterations), max(iterations))
        
        # Add average line
        avg_time = sum(iter_time) / len(iter_time)
        ax4.axhline(y=avg_time, color='red', linestyle='--', alpha=0.7, 
                   label=f'Average: {avg_time:.1f}s')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No timing data available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Iteration Time')
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Plot saved to: {save_path}")
    
    if show:
        plt.show()


def print_summary(history: list):
    """Print a text summary of training progress."""
    print("\n" + "=" * 60)
    print("📊 TRAINING SUMMARY")
    print("=" * 60)
    
    first = history[0]
    last = history[-1]
    
    print(f"\nIterations: {first['iteration']} → {last['iteration']} ({len(history)} recorded)")
    print(f"Games played: {first['games_played']} → {last['games_played']}")
    print(f"Train steps: {first['train_steps']} → {last['train_steps']}")
    
    print(f"\n📈 Loss Progress:")
    print(f"   Policy:  {first['policy_loss']:.4f} → {last['policy_loss']:.4f} ({last['policy_loss'] - first['policy_loss']:+.4f})")
    print(f"   Value:   {first['value_loss']:.4f} → {last['value_loss']:.4f} ({last['value_loss'] - first['value_loss']:+.4f})")
    print(f"   Total:   {first['total_loss']:.4f} → {last['total_loss']:.4f} ({last['total_loss'] - first['total_loss']:+.4f})")
    
    # Find min losses
    min_policy = min(h['policy_loss'] for h in history)
    min_value = min(h['value_loss'] for h in history)
    min_total = min(h['total_loss'] for h in history)
    
    min_policy_iter = next(h['iteration'] for h in history if h['policy_loss'] == min_policy)
    min_value_iter = next(h['iteration'] for h in history if h['value_loss'] == min_value)
    min_total_iter = next(h['iteration'] for h in history if h['total_loss'] == min_total)
    
    print(f"\n🏆 Best Losses:")
    print(f"   Policy:  {min_policy:.4f} (iter {min_policy_iter})")
    print(f"   Value:   {min_value:.4f} (iter {min_value_iter})")
    print(f"   Total:   {min_total:.4f} (iter {min_total_iter})")
    
    # Timing
    if history[0].get('iteration_time'):
        times = [h['iteration_time'] for h in history]
        avg_time = sum(times) / len(times)
        total_time = last.get('total_time', sum(times))
        print(f"\n⏱️ Timing:")
        print(f"   Average iteration: {avg_time:.1f}s")
        print(f"   Total training time: {total_time/3600:.1f}h")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Azul RL training history",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--history", "-f",
        type=str,
        default="checkpoints/training_history.json",
        help="Path to training history JSON file"
    )
    parser.add_argument(
        "--save", "-s",
        type=str,
        default=None,
        help="Save plot to file (e.g., training_plot.png)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot (useful with --save)"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print text summary, no plot"
    )
    
    args = parser.parse_args()
    
    # Load history
    history = load_history(args.history)
    print(f"📂 Loaded {len(history)} training entries from {args.history}")
    
    # Print summary
    print_summary(history)
    
    # Plot if requested
    if not args.summary_only:
        show = not args.no_show
        plot_training(history, save_path=args.save, show=show)


if __name__ == "__main__":
    main()

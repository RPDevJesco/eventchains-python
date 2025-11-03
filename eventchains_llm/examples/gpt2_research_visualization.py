"""
GPT-2 Research Insights Visualization

This script extends the research insights example with matplotlib visualizations
to create publication-quality figures showing:
- Token probability trajectories across layers
- Competitive token dynamics
- Confidence evolution
- Layer contribution heatmaps
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn.functional as F
from collections import defaultdict
import numpy as np

from eventchains import EventChain, ChainableEvent, EventContext, Result, Middleware

# Import from the base research script
from gpt2_research_insights import (
    ensure_dependencies,
    download_model,
    create_research_chain
)


def ensure_viz_dependencies():
    """Ensure matplotlib is installed."""
    print("Checking visualization dependencies...")
    
    try:
        import matplotlib
        print("✓ matplotlib installed")
    except ImportError:
        print("Installing matplotlib...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
        print("✓ matplotlib installed")
    
    print()


def plot_token_trajectories(trajectories, tokenizer, tracked_ids, output_path='token_trajectories.png'):
    """Plot how token probabilities evolve through layers."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for idx, token_id in enumerate(tracked_ids[:5]):
        token = tokenizer.decode([token_id])
        trajectory = trajectories.get(token_id, [])
        
        if trajectory:
            layers = [point['layer'] for point in trajectory]
            probs = [point['probability'] for point in trajectory]
            
            ax.plot(layers, probs, marker='o', linewidth=2, markersize=8,
                   label=f"'{token}' (ID: {token_id})", color=colors[idx])
    
    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Probability', fontsize=14, fontweight='bold')
    ax.set_title('Token Probability Evolution Across Layers', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(range(6))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_confidence_evolution(confidence_data, output_path='confidence_evolution.png'):
    """Plot model confidence metrics across layers."""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    layers = [c['layer'] for c in confidence_data]
    max_probs = [c['max_probability'] for c in confidence_data]
    entropies = [c['entropy'] for c in confidence_data]
    top5_mass = [c['top5_mass'] for c in confidence_data]
    
    # Plot 1: Max Probability and Top-5 Mass
    ax1.plot(layers, max_probs, marker='o', linewidth=2.5, markersize=9,
            label='Max Probability', color='#2E86AB')
    ax1.plot(layers, top5_mass, marker='s', linewidth=2.5, markersize=9,
            label='Top-5 Probability Mass', color='#A23B72')
    ax1.set_ylabel('Probability', fontsize=13, fontweight='bold')
    ax1.set_title('Model Certainty Across Layers', fontsize=15, fontweight='bold', pad=15)
    ax1.legend(fontsize=11, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(range(6))
    ax1.set_ylim([0, 1])
    
    # Plot 2: Entropy (uncertainty)
    ax2.plot(layers, entropies, marker='D', linewidth=2.5, markersize=9,
            label='Entropy (Uncertainty)', color='#F18F01')
    ax2.set_xlabel('Layer', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Entropy', fontsize=13, fontweight='bold')
    ax2.set_title('Model Uncertainty Across Layers', fontsize=15, fontweight='bold', pad=15)
    ax2.legend(fontsize=11, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xticks(range(6))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_competitive_dynamics(competition_data, tokenizer, selected_token_id, output_path='competitive_dynamics.png'):
    """Plot the probability gap between winner and runner-up across layers."""
    import matplotlib.pyplot as plt
    
    # Filter for competitions involving the selected token
    layer_gaps = defaultdict(list)
    for comp in competition_data:
        if comp['rank_1_id'] == selected_token_id:
            layer_gaps[comp['layer']].append(comp)
    
    layers = sorted(layer_gaps.keys())
    gaps = []
    winner_probs = []
    runner_probs = []
    
    for layer in layers:
        comps = layer_gaps[layer]
        if comps:
            comp = comps[0]
            gaps.append(comp['probability_gap'])
            winner_probs.append(comp['rank_1_prob'])
            runner_probs.append(comp['rank_2_prob'])
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.array(layers)
    ax.plot(x, winner_probs, marker='o', linewidth=2.5, markersize=9,
           label='Winner Probability', color='#06A77D')
    ax.plot(x, runner_probs, marker='s', linewidth=2.5, markersize=9,
           label='Runner-up Probability', color='#D62246')
    
    # Fill the gap area
    ax.fill_between(x, winner_probs, runner_probs, alpha=0.3, color='#FFB81C',
                    label='Probability Gap')
    
    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Probability', fontsize=14, fontweight='bold')
    ax.set_title('Competitive Token Dynamics: Winner vs Runner-up', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(range(6))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_layer_contributions(contributions, output_path='layer_contributions.png'):
    """Plot layer-wise contribution metrics."""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    layers = [c['layer'] for c in contributions]
    delta_norms = [c['delta_norm'] for c in contributions]
    prob_changes = [c['selected_token_prob_change'] for c in contributions]
    kl_divs = [c['distribution_shift_kl'] for c in contributions]
    
    # Plot 1: Probability Change (most important)
    colors_1 = ['#06A77D' if x > 0 else '#D62246' for x in prob_changes]
    bars1 = ax1.bar(layers, prob_changes, color=colors_1, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_xlabel('Layer', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Probability Change', fontsize=13, fontweight='bold')
    ax1.set_title('Layer Impact on Selected Token Probability', 
                 fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax1.set_xticks(range(6))
    
    # Add value labels on bars
    for bar, val in zip(bars1, prob_changes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:+.4f}', ha='center', va='bottom' if val > 0 else 'top',
                fontsize=9, fontweight='bold')
    
    # Plot 2: Distribution Shift (KL Divergence)
    bars2 = ax2.bar(layers, kl_divs, color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Layer', fontsize=13, fontweight='bold')
    ax2.set_ylabel('KL Divergence', fontsize=13, fontweight='bold')
    ax2.set_title('Distribution Shift per Layer', 
                 fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.set_xticks(range(6))
    
    # Add value labels on bars
    for bar, val in zip(bars2, kl_divs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_rank_evolution(trajectories, tokenizer, tracked_ids, output_path='rank_evolution.png'):
    """Plot how token ranks change through layers."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for idx, token_id in enumerate(tracked_ids[:5]):
        token = tokenizer.decode([token_id])
        trajectory = trajectories.get(token_id, [])
        
        if trajectory:
            layers = [point['layer'] for point in trajectory]
            ranks = [point['rank'] for point in trajectory]
            
            ax.plot(layers, ranks, marker='o', linewidth=2, markersize=8,
                   label=f"'{token}'", color=colors[idx])
    
    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Rank (lower is better)', fontsize=14, fontweight='bold')
    ax.set_title('Token Rank Evolution Across Layers', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(range(6))
    ax.invert_yaxis()  # Lower rank = better, so invert y-axis
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    print("=" * 80)
    print("EventChains LLM Research Insights + Visualization")
    print("=" * 80)
    print()
    
    ensure_dependencies()
    ensure_viz_dependencies()
    
    model, tokenizer = download_model()
    
    print("=" * 80)
    print("Building Research Chain")
    print("=" * 80)
    
    chain, middleware_dict = create_research_chain(model)
    
    print(f"✓ Research chain built")
    print()
    
    print("=" * 80)
    print("Running Inference")
    print("=" * 80)
    
    input_text = "What are the benefits of using Rust over languages like Go, C or C++ that aren't memory safety related?"
    print(f"Input: \"{input_text}\"")
    print()
    
    context = EventContext({
        'input_text': input_text,
        'model': model,
        'tokenizer': tokenizer
    })
    
    result = chain.execute(context)
    
    if not result.success:
        print(f"✗ Inference failed: {result.error}")
        return
    
    print("✓ Inference complete")
    print()
    
    # Extract data
    trajectories = middleware_dict['trajectory'].get_summary()
    competition_data = middleware_dict['competitive'].get_summary()
    confidence_data = middleware_dict['confidence'].get_summary()
    contributions = middleware_dict['contribution'].get_summary()
    
    tracked_ids = context.get('tracked_token_ids', [])
    selected_token_id = context.get('selected_token_id')
    
    # Generate visualizations
    print("=" * 80)
    print("Generating Visualizations")
    print("=" * 80)
    print()
    
    plot_token_trajectories(trajectories, tokenizer, tracked_ids)
    plot_confidence_evolution(confidence_data)
    plot_competitive_dynamics(competition_data, tokenizer, selected_token_id)
    plot_layer_contributions(contributions)
    plot_rank_evolution(trajectories, tokenizer, tracked_ids)
    
    print()
    print("=" * 80)
    print("Visualization Complete!")
    print("=" * 80)
    print()
    print("Generated files:")
    print("  • token_trajectories.png - Probability evolution")
    print("  • confidence_evolution.png - Model certainty metrics")
    print("  • competitive_dynamics.png - Winner vs runner-up")
    print("  • layer_contributions.png - Layer importance")
    print("  • rank_evolution.png - Token ranking changes")
    print()
    print("These visualizations are publication-ready and reveal:")
    print("  ✓ How the model gradually converges on its decision")
    print("  ✓ Which layers contribute most to the final output")
    print("  ✓ Competitive dynamics between tokens")
    print("  ✓ Model confidence progression")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()

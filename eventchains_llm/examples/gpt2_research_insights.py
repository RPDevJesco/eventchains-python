"""
GPT-2 Advanced Research Insights with EventChains

This example demonstrates deep research capabilities that are impossible with standard
Hugging Face inference:

1. Token Probability Trajectories - Track how probabilities evolve layer-by-layer
2. Alternative Token Suppression - Why specific tokens lose to others
3. Confidence Evolution - Model certainty progression through layers
4. Attention Flow Analysis - Token influence pathways
5. Layer Contribution Attribution - Which layers matter most for the decision
6. Competitive Token Analysis - Head-to-head probability races

These insights are crucial for:
- Mechanistic interpretability research
- Model debugging and safety analysis
- Understanding training dynamics
- Architecture improvements
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn.functional as F
from collections import defaultdict
import numpy as np

from eventchains import EventChain, ChainableEvent, EventContext, Result, Middleware


def ensure_dependencies():
    """Ensure required packages are installed."""
    print("Checking dependencies...")
    
    try:
        import transformers
        print("✓ transformers installed")
    except ImportError:
        print("Installing transformers...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
        print("✓ transformers installed")
    
    print()


def download_model():
    """Download DistilGPT-2 model and tokenizer."""
    print("=" * 80)
    print("Downloading DistilGPT-2 Model")
    print("=" * 80)
    print("Model size: ~350MB")
    print("This will be cached for future runs")
    print()
    
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    print("Downloading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    print("✓ Tokenizer ready")
    
    print("\nDownloading model (this may take a minute)...")
    model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    model.eval()
    print("✓ Model ready")
    print()
    
    return model, tokenizer


# ============================================================================
# Core Events for LLM Inference
# ============================================================================

class TokenizeInputEvent(ChainableEvent):
    """Convert input text to token IDs."""
    
    def execute(self, context):
        tokenizer = context.get('tokenizer')
        input_text = context.get('input_text')
        
        if not input_text:
            return Result.fail("No input text provided")
        
        inputs = tokenizer(input_text, return_tensors='pt')
        input_ids = inputs['input_ids']
        
        context.set('input_ids', input_ids)
        context.set('input_tokens', tokenizer.convert_ids_to_tokens(input_ids[0]))
        
        return Result.ok()


class EmbeddingEvent(ChainableEvent):
    """Convert token IDs to embeddings."""
    
    def execute(self, context):
        model = context.get('model')
        input_ids = context.get('input_ids')
        
        with torch.no_grad():
            embeddings = model.transformer.wte(input_ids)
            position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long)
            position_embeds = model.transformer.wpe(position_ids)
            hidden_states = embeddings + position_embeds
        
        context.set('hidden_states', hidden_states)
        context.set('embedding_norm', hidden_states.norm().item())
        
        return Result.ok()


class TransformerLayerEvent(ChainableEvent):
    """Process one transformer layer with detailed instrumentation."""
    
    def __init__(self, layer_idx, layer_module):
        self.layer_idx = layer_idx
        self.layer = layer_module
    
    def execute(self, context):
        hidden_states = context.get('hidden_states')
        model = context.get('model')
        
        # Store input to this layer for contribution analysis
        layer_input = hidden_states.clone()
        context.set(f'layer_{self.layer_idx}_input', layer_input)
        
        # Store current layer for middleware
        context.set('current_layer_idx', self.layer_idx)
        
        # Process through the layer
        with torch.no_grad():
            # Layer norm before attention
            ln1_output = self.layer.ln_1(hidden_states)
            
            # Attention
            attn_output = self.layer.attn(ln1_output)[0]
            
            # Residual connection after attention
            hidden_states = hidden_states + attn_output
            attn_residual = hidden_states.clone()
            context.set(f'layer_{self.layer_idx}_after_attention', attn_residual)
            
            # Layer norm before MLP
            ln2_output = self.layer.ln_2(hidden_states)
            
            # MLP
            mlp_output = self.layer.mlp(ln2_output)
            
            # Residual connection after MLP
            hidden_states = hidden_states + mlp_output
        
        context.set('hidden_states', hidden_states)
        context.set(f'layer_{self.layer_idx}_output', hidden_states.clone())
        
        # Compute intermediate logits for this layer (for trajectory tracking)
        with torch.no_grad():
            # Apply final layer norm and get logits
            ln_f_output = model.transformer.ln_f(hidden_states)
            intermediate_logits = model.lm_head(ln_f_output)
            context.set(f'layer_{self.layer_idx}_logits', intermediate_logits)
        
        return Result.ok()


class FinalLayerNormEvent(ChainableEvent):
    """Apply final layer normalization."""
    
    def execute(self, context):
        model = context.get('model')
        hidden_states = context.get('hidden_states')
        
        with torch.no_grad():
            hidden_states = model.transformer.ln_f(hidden_states)
        
        context.set('hidden_states', hidden_states)
        return Result.ok()


class LogitComputationEvent(ChainableEvent):
    """Compute final logits for next token prediction."""
    
    def execute(self, context):
        model = context.get('model')
        hidden_states = context.get('hidden_states')
        
        with torch.no_grad():
            logits = model.lm_head(hidden_states)
        
        context.set('logits', logits)
        context.set('final_logits', logits.clone())
        
        # Compute statistics
        last_token_logits = logits[0, -1, :]
        probs = F.softmax(last_token_logits, dim=-1)
        
        top_k = 10
        top_probs, top_indices = probs.topk(top_k)
        
        context.set('top_k_probs', top_probs)
        context.set('top_k_indices', top_indices)
        context.set('final_probs', probs)
        
        return Result.ok()


class TokenSelectionEvent(ChainableEvent):
    """Select the next token (greedy decoding)."""
    
    def execute(self, context):
        tokenizer = context.get('tokenizer')
        logits = context.get('logits')
        
        next_token_id = logits[0, -1, :].argmax().item()
        next_token = tokenizer.decode([next_token_id])
        
        context.set('selected_token_id', next_token_id)
        context.set('selected_token', next_token)
        
        return Result.ok()


# ============================================================================
# Research Middleware - Advanced Analysis
# ============================================================================

class TokenProbabilityTrajectoryMiddleware(Middleware):
    """
    Track how token probabilities evolve through each layer.
    
    This answers: "How did the model gradually converge on its final answer?"
    """
    
    def __init__(self, top_k=5):
        self.top_k = top_k
        self.trajectories = defaultdict(list)
        self.layer_count = 0
    
    def execute(self, context, next_callable):
        result = next_callable(context)
        
        event_name = context.get('_current_event', '')
        if 'TransformerLayer' in event_name:
            layer_idx = context.get('current_layer_idx')
            logits = context.get(f'layer_{layer_idx}_logits')
            
            if logits is not None:
                probs = F.softmax(logits[0, -1, :], dim=-1)
                
                # Get top-k token IDs from final prediction
                final_probs = context.get('final_probs')
                if final_probs is None:
                    # If final probs not available yet, use current top-k
                    top_probs, top_indices = probs.topk(self.top_k)
                    context.set('tracked_token_ids', top_indices.tolist())
                else:
                    tracked_ids = context.get('tracked_token_ids')
                    if tracked_ids is None:
                        _, top_indices = final_probs.topk(self.top_k)
                        tracked_ids = top_indices.tolist()
                        context.set('tracked_token_ids', tracked_ids)
                    
                    # Track probabilities for these specific tokens
                    for token_id in tracked_ids:
                        prob = probs[token_id].item()
                        self.trajectories[token_id].append({
                            'layer': layer_idx,
                            'probability': prob,
                            'rank': self._get_rank(probs, token_id)
                        })
                
                self.layer_count = max(self.layer_count, layer_idx + 1)
        
        return result
    
    def _get_rank(self, probs, token_id):
        """Get the rank of a token in the probability distribution."""
        sorted_probs, sorted_indices = probs.sort(descending=True)
        rank = (sorted_indices == token_id).nonzero(as_tuple=True)[0].item()
        return rank + 1
    
    def get_summary(self):
        return dict(self.trajectories)


class CompetitiveTokenAnalysisMiddleware(Middleware):
    """
    Analyze why the model chose token A over token B.
    
    This answers: "What caused the winning token to suppress the alternatives?"
    """
    
    def __init__(self, num_competitors=3):
        self.num_competitors = num_competitors
        self.competition_data = []
    
    def execute(self, context, next_callable):
        result = next_callable(context)
        
        event_name = context.get('_current_event', '')
        if 'TransformerLayer' in event_name:
            layer_idx = context.get('current_layer_idx')
            logits = context.get(f'layer_{layer_idx}_logits')
            
            if logits is not None:
                probs = F.softmax(logits[0, -1, :], dim=-1)
                top_probs, top_indices = probs.topk(self.num_competitors + 1)
                
                # Calculate probability gaps between consecutive tokens
                gaps = []
                for i in range(len(top_probs) - 1):
                    gap = (top_probs[i] - top_probs[i + 1]).item()
                    gaps.append({
                        'layer': layer_idx,
                        'rank_1_id': top_indices[i].item(),
                        'rank_2_id': top_indices[i + 1].item(),
                        'probability_gap': gap,
                        'rank_1_prob': top_probs[i].item(),
                        'rank_2_prob': top_probs[i + 1].item()
                    })
                
                self.competition_data.extend(gaps)
        
        return result
    
    def get_summary(self):
        return self.competition_data


class ConfidenceEvolutionMiddleware(Middleware):
    """
    Track model confidence evolution through layers.
    
    This answers: "How certain is the model, and when did it become certain?"
    """
    
    def __init__(self):
        self.confidence_data = []
    
    def execute(self, context, next_callable):
        result = next_callable(context)
        
        event_name = context.get('_current_event', '')
        if 'TransformerLayer' in event_name:
            layer_idx = context.get('current_layer_idx')
            logits = context.get(f'layer_{layer_idx}_logits')
            
            if logits is not None:
                probs = F.softmax(logits[0, -1, :], dim=-1)
                
                # Calculate entropy (lower = more confident)
                entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
                
                # Get max probability (higher = more confident)
                max_prob = probs.max().item()
                
                # Calculate top-5 probability mass
                top5_probs = probs.topk(5)[0]
                top5_mass = top5_probs.sum().item()
                
                self.confidence_data.append({
                    'layer': layer_idx,
                    'entropy': entropy,
                    'max_probability': max_prob,
                    'top5_mass': top5_mass,
                    'confidence_score': max_prob / entropy if entropy > 0 else 0
                })
        
        return result
    
    def get_summary(self):
        return self.confidence_data


class LayerContributionMiddleware(Middleware):
    """
    Attribute the final prediction to specific layers.
    
    This answers: "Which layers were most important for this decision?"
    """
    
    def __init__(self):
        self.contributions = []
    
    def execute(self, context, next_callable):
        result = next_callable(context)
        
        event_name = context.get('_current_event', '')
        if 'TransformerLayer' in event_name:
            layer_idx = context.get('current_layer_idx')
            
            layer_input = context.get(f'layer_{layer_idx}_input')
            layer_output = context.get(f'layer_{layer_idx}_output')
            
            if layer_input is not None and layer_output is not None:
                # Calculate change in hidden states
                delta = layer_output - layer_input
                delta_norm = delta.norm().item()
                
                # Get logits before and after this layer
                model = context.get('model')
                with torch.no_grad():
                    ln_f = model.transformer.ln_f
                    lm_head = model.lm_head
                    
                    logits_before = lm_head(ln_f(layer_input))
                    logits_after = lm_head(ln_f(layer_output))
                    
                    # Get probabilities
                    probs_before = F.softmax(logits_before[0, -1, :], dim=-1)
                    probs_after = F.softmax(logits_after[0, -1, :], dim=-1)
                    
                    # Get the final selected token ID
                    selected_id = context.get('selected_token_id')
                    if selected_id is None:
                        # Use current top prediction
                        selected_id = probs_after.argmax().item()
                    
                    prob_change = probs_after[selected_id].item() - probs_before[selected_id].item()
                    
                    # Calculate KL divergence to measure distribution shift
                    kl_div = F.kl_div(
                        probs_before.log(),
                        probs_after,
                        reduction='sum'
                    ).item()
                
                self.contributions.append({
                    'layer': layer_idx,
                    'delta_norm': delta_norm,
                    'selected_token_prob_change': prob_change,
                    'distribution_shift_kl': kl_div
                })
        
        return result
    
    def get_summary(self):
        return self.contributions


# ============================================================================
# Main Demo
# ============================================================================

def create_research_chain(model):
    """Create EventChain with advanced research instrumentation."""
    
    # Initialize research middleware
    trajectory_tracker = TokenProbabilityTrajectoryMiddleware(top_k=5)
    competitive_analyzer = CompetitiveTokenAnalysisMiddleware(num_competitors=3)
    confidence_tracker = ConfidenceEvolutionMiddleware()
    contribution_analyzer = LayerContributionMiddleware()
    
    # Build chain with all 6 transformer layers
    chain = EventChain()
    chain.add_event(TokenizeInputEvent())
    chain.add_event(EmbeddingEvent())
    
    for layer_idx in range(6):
        chain.add_event(TransformerLayerEvent(layer_idx, model.transformer.h[layer_idx]))
    
    chain.add_event(FinalLayerNormEvent())
    chain.add_event(LogitComputationEvent())
    chain.add_event(TokenSelectionEvent())
    
    # Add research middleware
    chain.use_middleware(trajectory_tracker)
    chain.use_middleware(competitive_analyzer)
    chain.use_middleware(confidence_tracker)
    chain.use_middleware(contribution_analyzer)
    
    return chain, {
        'trajectory': trajectory_tracker,
        'competitive': competitive_analyzer,
        'confidence': confidence_tracker,
        'contribution': contribution_analyzer
    }


def print_research_results(context, middleware_dict, tokenizer):
    """Print all research insights."""
    
    print("\n" + "=" * 80)
    print("EVENTCHAINS LLM RESEARCH INSIGHTS")
    print("=" * 80)
    
    # Input/Output
    print("\n1. BASIC INFERENCE")
    print("-" * 80)
    input_text = context.get('input_text')
    selected_token = context.get('selected_token')
    print(f"Input: \"{input_text}\"")
    print(f"Predicted next token: '{selected_token}'")
    
    # Top predictions
    print("\nTop-5 Final Predictions:")
    top_k_indices = context.get('top_k_indices')
    top_k_probs = context.get('top_k_probs')
    for idx, (token_id, prob) in enumerate(zip(top_k_indices[:5], top_k_probs[:5]), 1):
        token = tokenizer.decode([token_id.item()])
        print(f"  {idx}. '{token}' (prob: {prob.item():.4f})")
    
    # Token Probability Trajectories
    print("\n" + "=" * 80)
    print("2. TOKEN PROBABILITY TRAJECTORIES")
    print("-" * 80)
    print("How did probabilities evolve layer-by-layer?\n")
    
    trajectories = middleware_dict['trajectory'].get_summary()
    tracked_ids = context.get('tracked_token_ids', [])
    
    for token_id in tracked_ids[:5]:
        token = tokenizer.decode([token_id])
        trajectory = trajectories.get(token_id, [])
        
        if trajectory:
            print(f"Token: '{token}' (ID: {token_id})")
            print(f"{'Layer':<8} {'Probability':<15} {'Rank':<10} {'Change':<15}")
            print("-" * 60)
            
            for i, point in enumerate(trajectory):
                prob = point['probability']
                rank = point['rank']
                
                if i > 0:
                    prev_prob = trajectory[i-1]['probability']
                    change = prob - prev_prob
                    change_str = f"{change:+.6f}"
                else:
                    change_str = "—"
                
                print(f"{point['layer']:<8} {prob:<15.6f} {rank:<10} {change_str:<15}")
            
            print()
    
    # Competitive Token Analysis
    print("=" * 80)
    print("3. COMPETITIVE TOKEN ANALYSIS")
    print("-" * 80)
    print("Why did winning tokens suppress alternatives?\n")
    
    competition = middleware_dict['competitive'].get_summary()
    
    # Get the competition for rank 1 vs rank 2 at each layer
    print(f"{'Layer':<8} {'Winner':<20} {'Runner-up':<20} {'Gap':<12} {'Winner %':<12}")
    print("-" * 80)
    
    layer_competitions = defaultdict(list)
    for comp in competition:
        if comp['rank_1_id'] == context.get('selected_token_id'):
            layer_competitions[comp['layer']].append(comp)
    
    for layer_idx in sorted(layer_competitions.keys()):
        comps = layer_competitions[layer_idx]
        if comps:
            comp = comps[0]  # Take the first competition (rank 1 vs rank 2)
            winner_token = tokenizer.decode([comp['rank_1_id']])
            runner_token = tokenizer.decode([comp['rank_2_id']])
            gap = comp['probability_gap']
            winner_prob = comp['rank_1_prob']
            
            print(f"{layer_idx:<8} '{winner_token}'".ljust(28) + 
                  f"'{runner_token}'".ljust(28) + 
                  f"{gap:<12.6f} {winner_prob:<12.4f}")
    
    # Confidence Evolution
    print("\n" + "=" * 80)
    print("4. CONFIDENCE EVOLUTION")
    print("-" * 80)
    print("How did model certainty change through layers?\n")
    
    confidence = middleware_dict['confidence'].get_summary()
    
    print(f"{'Layer':<8} {'Max Prob':<12} {'Entropy':<12} {'Top-5 Mass':<12} {'Confidence':<12}")
    print("-" * 70)
    
    for conf in confidence:
        print(f"{conf['layer']:<8} "
              f"{conf['max_probability']:<12.4f} "
              f"{conf['entropy']:<12.4f} "
              f"{conf['top5_mass']:<12.4f} "
              f"{conf['confidence_score']:<12.4f}")
    
    # Layer Contribution
    print("\n" + "=" * 80)
    print("5. LAYER CONTRIBUTION ATTRIBUTION")
    print("-" * 80)
    print("Which layers mattered most for the final decision?\n")
    
    contributions = middleware_dict['contribution'].get_summary()
    
    print(f"{'Layer':<8} {'Delta Norm':<15} {'Prob Change':<15} {'KL Divergence':<15}")
    print("-" * 60)
    
    for contrib in contributions:
        print(f"{contrib['layer']:<8} "
              f"{contrib['delta_norm']:<15.6f} "
              f"{contrib['selected_token_prob_change']:<+15.6f} "
              f"{contrib['distribution_shift_kl']:<15.6f}")
    
    # Find most influential layer
    if contributions:
        max_contrib = max(contributions, key=lambda x: abs(x['selected_token_prob_change']))
        print(f"\n💡 Most influential layer: Layer {max_contrib['layer']} "
              f"(changed selected token prob by {max_contrib['selected_token_prob_change']:+.6f})")
    
    print("\n" + "=" * 80)
    print("RESEARCH INSIGHTS SUMMARY")
    print("=" * 80)
    print("\nStandard Hugging Face provides:")
    print("  ✓ Final output only")
    print()
    print("EventChains ADDITIONALLY provides:")
    print("  ✓ Token probability evolution through all layers")
    print("  ✓ Competitive dynamics between tokens")
    print("  ✓ Model confidence progression")
    print("  ✓ Layer-specific contribution attribution")
    print("  ✓ Decision pathway visualization")
    print()
    print("These insights enable:")
    print("  • Mechanistic interpretability research")
    print("  • Model safety and alignment analysis")
    print("  • Architecture optimization")
    print("  • Training dynamics understanding")
    print()
    print("=" * 80)


def main():
    print("=" * 80)
    print("EventChains LLM Research Insights Demo")
    print("Advanced Analysis for LLM Decision-Making")
    print("=" * 80)
    print()
    
    ensure_dependencies()
    model, tokenizer = download_model()
    
    print("=" * 80)
    print("Building Research-Grade Instrumentation Chain")
    print("=" * 80)
    print("Creating EventChain with:")
    print("  - Token probability trajectory tracking")
    print("  - Competitive token analysis")
    print("  - Confidence evolution monitoring")
    print("  - Layer contribution attribution")
    print()
    
    chain, middleware_dict = create_research_chain(model)
    
    print(f"✓ Research chain built: {chain}")
    print()
    
    print("=" * 80)
    print("Running Inference with Full Research Instrumentation")
    print("=" * 80)
    
    input_text = "What are the benefits of using Rust over languages like Go, C or C++ that aren't memory safety related?"
    print(f"Input: \"{input_text}\"")
    print()
    print("Executing chain with deep analysis...")
    
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
    
    print_research_results(context, middleware_dict, tokenizer)
    
    print("\n" + "=" * 80)
    print("EventChains: Enabling LLM Research")
    print("=" * 80)


if __name__ == "__main__":
    main()

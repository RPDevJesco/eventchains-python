"""
GPT-2 Explainability with EventChains

Demonstrates data capture that standard Hugging Face inference misses:
1. K/Q/V matrix norms per layer
2. MLP intermediate activation sparsity
3. Layer normalization statistics
4. Residual stream decomposition
5. Attention head behavioral classification
6. Real-time numerical validation

First run will download DistilGPT-2 (~350MB) - subsequent runs use cached model.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn.functional as F
from collections import defaultdict

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
# EventChains Events for LLM Inference
# ============================================================================

class TokenizeInputEvent(ChainableEvent):
    """Convert input text to token IDs."""

    def execute(self, context):
        tokenizer = context.get('tokenizer')
        input_text = context.get('input_text')

        if not input_text:
            return Result.fail("No input text provided")

        # Tokenize
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

        # Get embeddings (token + position embeddings)
        with torch.no_grad():
            embeddings = model.transformer.wte(input_ids)  # Word token embeddings
            position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long)
            position_embeds = model.transformer.wpe(position_ids)  # Position embeddings

            hidden_states = embeddings + position_embeds

        context.set('hidden_states', hidden_states)
        context.set('embedding_norm', hidden_states.norm().item())

        return Result.ok()


class TransformerLayerEvent(ChainableEvent):
    """Process one transformer layer with full instrumentation."""

    def __init__(self, layer_idx, layer_module):
        self.layer_idx = layer_idx
        self.layer = layer_module

    def execute(self, context):
        hidden_states = context.get('hidden_states')

        # Store current layer for middleware
        context.set('current_layer_idx', self.layer_idx)

        # === LAYER NORM 1 (before attention) ===
        ln1_output = self.layer.ln_1(hidden_states)
        context.set(f'layer_{self.layer_idx}_ln1_mean', ln1_output.mean().item())
        context.set(f'layer_{self.layer_idx}_ln1_std', ln1_output.std().item())

        # === ATTENTION ===
        with torch.no_grad():
            # Get attention output and weights
            attn_output = self.layer.attn(ln1_output)

            # Manual K/Q/V computation to capture matrices
            c_attn = self.layer.attn.c_attn
            qkv = c_attn(ln1_output)

            # Split into Q, K, V
            query, key, value = qkv.split(self.layer.attn.split_size, dim=2)

            # Reshape for multi-head attention
            query = self._split_heads(query, self.layer.attn.num_heads, self.layer.attn.head_dim)
            key = self._split_heads(key, self.layer.attn.num_heads, self.layer.attn.head_dim)
            value = self._split_heads(value, self.layer.attn.num_heads, self.layer.attn.head_dim)

            # Compute attention weights
            attn_weights = torch.matmul(query, key.transpose(-1, -2))
            attn_weights = attn_weights / torch.sqrt(torch.tensor(query.size(-1), dtype=torch.float32))
            attn_weights = F.softmax(attn_weights, dim=-1)

            # Store K/Q/V norms
            context.set(f'layer_{self.layer_idx}_kqv_norms', {
                'key': key.norm().item(),
                'query': query.norm().item(),
                'value': value.norm().item()
            })

            # Store attention weights and analysis
            context.set(f'layer_{self.layer_idx}_attention_weights', attn_weights)
            context.set(f'layer_{self.layer_idx}_attention_entropy',
                        self._compute_attention_entropy(attn_weights))
            context.set(f'layer_{self.layer_idx}_head_behaviors',
                        self._classify_head_behaviors(attn_weights))

        # Add residual connection
        hidden_states = hidden_states + attn_output[0]
        attn_residual_norm = attn_output[0].norm().item()

        # === LAYER NORM 2 (before MLP) ===
        ln2_output = self.layer.ln_2(hidden_states)

        # === MLP ===
        with torch.no_grad():
            mlp_fc_output = self.layer.mlp.c_fc(ln2_output)  # First linear layer
            mlp_activated = self.layer.mlp.act(mlp_fc_output)  # GELU activation
            mlp_proj_output = self.layer.mlp.c_proj(mlp_activated)  # Second linear layer

            # Store MLP statistics
            sparsity = (mlp_activated.abs() < 1e-3).float().mean().item()
            context.set(f'layer_{self.layer_idx}_mlp_sparsity', sparsity)
            context.set(f'layer_{self.layer_idx}_mlp_mean_activation', mlp_activated.mean().item())
            context.set(f'layer_{self.layer_idx}_mlp_output_norm', mlp_proj_output.norm().item())

        # Add residual connection
        hidden_states = hidden_states + mlp_proj_output
        mlp_residual_norm = mlp_proj_output.norm().item()

        # === RESIDUAL STREAM DECOMPOSITION ===
        total_residual = attn_residual_norm + mlp_residual_norm
        context.set(f'layer_{self.layer_idx}_residual_decomposition', {
            'attention_contribution': attn_residual_norm / total_residual if total_residual > 0 else 0,
            'mlp_contribution': mlp_residual_norm / total_residual if total_residual > 0 else 0,
            'attention_norm': attn_residual_norm,
            'mlp_norm': mlp_residual_norm
        })

        # Update hidden states
        context.set('hidden_states', hidden_states)

        return Result.ok()

    def _split_heads(self, tensor, num_heads, head_dim):
        """Split tensor into multiple attention heads."""
        batch_size, seq_len, _ = tensor.size()
        tensor = tensor.view(batch_size, seq_len, num_heads, head_dim)
        return tensor.permute(0, 2, 1, 3)  # [batch, heads, seq, head_dim]

    def _compute_attention_entropy(self, attn_weights):
        """Compute entropy of attention distribution."""
        # attn_weights: [batch, heads, seq, seq]
        # Average over batch and heads, look at last token
        avg_attn = attn_weights.mean(dim=(0, 1))[-1, :]  # Last token's attention
        entropy = -(avg_attn * torch.log(avg_attn + 1e-10)).sum().item()
        return entropy

    def _classify_head_behaviors(self, attn_weights):
        """Classify behavior of each attention head."""
        # attn_weights: [batch, heads, seq, seq]
        behaviors = {}

        num_heads = attn_weights.shape[1]
        for head_idx in range(num_heads):
            head_attn = attn_weights[0, head_idx, -1, :]  # Last token's attention

            # Classify behavior
            max_attn = head_attn.max().item()
            max_pos = head_attn.argmax().item()
            std_attn = head_attn.std().item()

            if max_pos == 0:
                behavior = 'first_token_focus'
            elif max_pos == len(head_attn) - 2:
                behavior = 'previous_token'
            elif std_attn < 0.05:
                behavior = 'uniform_aggregation'
            elif max_attn > 0.7:
                behavior = 'position_specific'
            else:
                behavior = 'distributed'

            behaviors[f'H{head_idx}'] = behavior

        return behaviors


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
    """Compute logits for next token prediction."""

    def execute(self, context):
        model = context.get('model')
        hidden_states = context.get('hidden_states')

        with torch.no_grad():
            logits = model.lm_head(hidden_states)

        context.set('logits', logits)

        # Compute statistics on logits
        last_token_logits = logits[0, -1, :]
        probs = F.softmax(last_token_logits, dim=-1)

        top_k = 5
        top_probs, top_indices = probs.topk(top_k)

        context.set('top_k_probs', top_probs)
        context.set('top_k_indices', top_indices)
        context.set('logit_entropy', -(probs * torch.log(probs + 1e-10)).sum().item())

        return Result.ok()


class TokenSelectionEvent(ChainableEvent):
    """Select the next token (greedy decoding)."""

    def execute(self, context):
        tokenizer = context.get('tokenizer')
        logits = context.get('logits')

        # Greedy selection
        next_token_id = logits[0, -1, :].argmax().item()
        next_token = tokenizer.decode([next_token_id])

        context.set('selected_token_id', next_token_id)
        context.set('selected_token', next_token)

        return Result.ok()


# ============================================================================
# EventChains Middleware for LLM Analysis
# ============================================================================

class KQVNormTrackerMiddleware(Middleware):
    """Track K/Q/V matrix norms across all layers."""

    def __init__(self):
        self.kqv_norms = []

    def execute(self, context, next_callable):
        result = next_callable(context)

        # After each transformer layer, collect K/Q/V norms
        event_name = context.get('_current_event', '')
        if 'TransformerLayer' in event_name:
            layer_idx = context.get('current_layer_idx')
            kqv = context.get(f'layer_{layer_idx}_kqv_norms')
            if kqv:
                self.kqv_norms.append({
                    'layer': layer_idx,
                    **kqv
                })

        return result

    def get_summary(self):
        return self.kqv_norms


class MLPSparsityMiddleware(Middleware):
    """Track MLP activation sparsity across all layers."""

    def __init__(self):
        self.sparsity_data = []

    def execute(self, context, next_callable):
        result = next_callable(context)

        # After each transformer layer, collect MLP sparsity
        event_name = context.get('_current_event', '')
        if 'TransformerLayer' in event_name:
            layer_idx = context.get('current_layer_idx')
            sparsity = context.get(f'layer_{layer_idx}_mlp_sparsity')
            if sparsity is not None:
                self.sparsity_data.append({
                    'layer': layer_idx,
                    'sparsity': sparsity
                })

        return result

    def get_summary(self):
        return self.sparsity_data


class AttentionHeadAnalyzerMiddleware(Middleware):
    """Analyze attention head behaviors across all layers."""

    def __init__(self):
        self.head_behaviors = defaultdict(list)

    def execute(self, context, next_callable):
        result = next_callable(context)

        # After each transformer layer, collect head behaviors
        event_name = context.get('_current_event', '')
        if 'TransformerLayer' in event_name:
            layer_idx = context.get('current_layer_idx')
            behaviors = context.get(f'layer_{layer_idx}_head_behaviors')
            if behaviors:
                for head_id, behavior in behaviors.items():
                    self.head_behaviors[f'L{layer_idx}{head_id}'] = behavior

        return result

    def get_summary(self):
        return dict(self.head_behaviors)


class NumericalValidationMiddleware(Middleware):
    """Validate numerical stability during inference."""

    def __init__(self, strict=False, verbose=True):
        self.strict = strict
        self.verbose = verbose
        self.issues = []

    def execute(self, context, next_callable):
        # Pre-execution check
        if context.has('hidden_states'):
            hidden = context.get('hidden_states')

            if torch.isnan(hidden).any():
                issue = "NaN detected in hidden states"
                self.issues.append(issue)
                if self.verbose:
                    print(f"⚠️  {issue}")
                if self.strict:
                    return Result.fail(issue)

            if torch.isinf(hidden).any():
                issue = "Inf detected in hidden states"
                self.issues.append(issue)
                if self.verbose:
                    print(f"⚠️  {issue}")
                if self.strict:
                    return Result.fail(issue)

            # Check for extremely large values
            max_val = hidden.abs().max().item()
            if max_val > 1e4:
                issue = f"Large activation detected: {max_val:.2e}"
                self.issues.append(issue)
                if self.verbose:
                    print(f"⚠️  {issue}")

        result = next_callable(context)

        # Post-execution check
        event_name = context.get('_current_event', '')
        if 'TransformerLayer' in event_name:
            layer_idx = context.get('current_layer_idx')

            # Check attention entropy
            entropy = context.get(f'layer_{layer_idx}_attention_entropy')
            if entropy and entropy < 0.1:
                issue = f"Low attention entropy in layer {layer_idx}: {entropy:.3f}"
                self.issues.append(issue)
                if self.verbose:
                    print(f"⚠️  {issue}")

        return result

    def get_issues(self):
        return self.issues


class ResidualStreamAnalyzerMiddleware(Middleware):
    """Analyze residual stream contributions from attention vs MLP."""

    def __init__(self):
        self.residual_data = []

    def execute(self, context, next_callable):
        result = next_callable(context)

        # After each transformer layer, collect residual decomposition
        event_name = context.get('_current_event', '')
        if 'TransformerLayer' in event_name:
            layer_idx = context.get('current_layer_idx')
            decomp = context.get(f'layer_{layer_idx}_residual_decomposition')
            if decomp:
                self.residual_data.append({
                    'layer': layer_idx,
                    **decomp
                })

        return result

    def get_summary(self):
        return self.residual_data


# ============================================================================
# Main Demo
# ============================================================================

def create_instrumented_chain(model):
    """Create EventChain with full instrumentation for all 6 layers."""

    # Initialize middleware
    kqv_tracker = KQVNormTrackerMiddleware()
    mlp_sparsity = MLPSparsityMiddleware()
    head_analyzer = AttentionHeadAnalyzerMiddleware()
    validation = NumericalValidationMiddleware(strict=False, verbose=False)
    residual_analyzer = ResidualStreamAnalyzerMiddleware()

    # Build chain with all 6 transformer layers
    chain = EventChain()
    chain.add_event(TokenizeInputEvent())
    chain.add_event(EmbeddingEvent())

    # Add all 6 transformer layers
    for layer_idx in range(6):
        chain.add_event(TransformerLayerEvent(layer_idx, model.transformer.h[layer_idx]))

    chain.add_event(FinalLayerNormEvent())
    chain.add_event(LogitComputationEvent())
    chain.add_event(TokenSelectionEvent())

    # Add middleware
    chain.use_middleware(kqv_tracker)
    chain.use_middleware(mlp_sparsity)
    chain.use_middleware(head_analyzer)
    chain.use_middleware(validation)
    chain.use_middleware(residual_analyzer)

    return chain, {
        'kqv_tracker': kqv_tracker,
        'mlp_sparsity': mlp_sparsity,
        'head_analyzer': head_analyzer,
        'validation': validation,
        'residual_analyzer': residual_analyzer
    }


def print_results(context, middleware_dict):
    """Print all the captured data."""

    print("\n" + "=" * 80)
    print("EVENTCHAINS LLM EXPLAINABILITY RESULTS")
    print("=" * 80)

    # Input/Output
    print("\n1. INPUT/OUTPUT")
    print("-" * 80)
    input_tokens = context.get('input_tokens')
    selected_token = context.get('selected_token')
    print(f"Input tokens: {input_tokens}")
    print(f"Selected next token: '{selected_token}'")

    # Top-K predictions
    print("\nTop-5 Next Token Predictions:")
    tokenizer = context.get('tokenizer')
    top_k_indices = context.get('top_k_indices')
    top_k_probs = context.get('top_k_probs')
    for idx, (token_id, prob) in enumerate(zip(top_k_indices, top_k_probs), 1):
        token = tokenizer.decode([token_id.item()])
        print(f"  {idx}. '{token}' (prob: {prob.item():.4f})")

    # K/Q/V Norms (NEW - not in Hugging Face)
    print("\n" + "=" * 80)
    print("2. K/Q/V MATRIX NORMS PER LAYER (NEW!)")
    print("-" * 80)
    print("Standard Hugging Face: ❌ Not available")
    print("EventChains:           ✓ Captured\n")
    kqv_data = middleware_dict['kqv_tracker'].get_summary()
    print(f"{'Layer':<8} {'Query Norm':<15} {'Key Norm':<15} {'Value Norm':<15}")
    print("-" * 60)
    for item in kqv_data:
        print(f"{item['layer']:<8} {item['query']:<15.3f} {item['key']:<15.3f} {item['value']:<15.3f}")

    # MLP Sparsity (NEW - not in Hugging Face)
    print("\n" + "=" * 80)
    print("3. MLP INTERMEDIATE ACTIVATION SPARSITY (NEW!)")
    print("-" * 80)
    print("Standard Hugging Face: ❌ Not available")
    print("EventChains:           ✓ Captured\n")
    sparsity_data = middleware_dict['mlp_sparsity'].get_summary()
    print(f"{'Layer':<8} {'Sparsity':<15} {'Active Neurons':<20}")
    print("-" * 50)
    for item in sparsity_data:
        sparsity = item['sparsity']
        active_pct = (1 - sparsity) * 100
        print(f"{item['layer']:<8} {sparsity:<15.2%} {active_pct:<20.1f}%")

    # Layer Norm Statistics (NEW - not in Hugging Face)
    print("\n" + "=" * 80)
    print("4. LAYER NORMALIZATION STATISTICS (NEW!)")
    print("-" * 80)
    print("Standard Hugging Face: ❌ Not available")
    print("EventChains:           ✓ Captured\n")
    print(f"{'Layer':<8} {'LN1 Mean':<15} {'LN1 Std':<15}")
    print("-" * 45)
    for layer_idx in range(6):
        ln1_mean = context.get(f'layer_{layer_idx}_ln1_mean', 0)
        ln1_std = context.get(f'layer_{layer_idx}_ln1_std', 0)
        print(f"{layer_idx:<8} {ln1_mean:<15.6f} {ln1_std:<15.6f}")

    # Residual Stream Decomposition (NEW - not in Hugging Face)
    print("\n" + "=" * 80)
    print("5. RESIDUAL STREAM DECOMPOSITION (NEW!)")
    print("-" * 80)
    print("Standard Hugging Face: ❌ Not available")
    print("EventChains:           ✓ Captured\n")
    residual_data = middleware_dict['residual_analyzer'].get_summary()
    print(f"{'Layer':<8} {'Attention %':<15} {'MLP %':<15}")
    print("-" * 45)
    for item in residual_data:
        attn_pct = item['attention_contribution'] * 100
        mlp_pct = item['mlp_contribution'] * 100
        print(f"{item['layer']:<8} {attn_pct:<15.1f} {mlp_pct:<15.1f}")

    # Attention Head Behavioral Classification (NEW - not in Hugging Face)
    print("\n" + "=" * 80)
    print("6. ATTENTION HEAD BEHAVIORAL CLASSIFICATION (NEW!)")
    print("-" * 80)
    print("Standard Hugging Face: ❌ Not available")
    print("EventChains:           ✓ Captured\n")
    head_behaviors = middleware_dict['head_analyzer'].get_summary()

    # Group by behavior type
    behavior_groups = defaultdict(list)
    for head_id, behavior in head_behaviors.items():
        behavior_groups[behavior].append(head_id)

    print("Behavior Distribution:")
    for behavior, heads in sorted(behavior_groups.items()):
        print(f"  {behavior:<25} {len(heads):>3} heads: {', '.join(heads[:5])}")
        if len(heads) > 5:
            print(f"{'':>30} (and {len(heads) - 5} more)")

    # Real-time Validation (NEW - not in Hugging Face)
    print("\n" + "=" * 80)
    print("7. REAL-TIME NUMERICAL VALIDATION (NEW!)")
    print("-" * 80)
    print("Standard Hugging Face: ❌ Not available")
    print("EventChains:           ✓ Captured\n")
    issues = middleware_dict['validation'].get_issues()
    if issues:
        print(f"⚠️  {len(issues)} issue(s) detected:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ No numerical issues detected")
        print("   All values within normal ranges")

    print("\n" + "=" * 80)
    print("SUMMARY: Data Captured by EventChains vs Hugging Face")
    print("=" * 80)
    print()
    print("Standard Hugging Face output_attentions=True provides:")
    print("  ✓ Final attention weights")
    print("  ✓ Final hidden states")
    print("  ✓ Logits")
    print()
    print("EventChains ADDITIONALLY provides:")
    print("  ✓ K/Q/V matrix norms per layer")
    print("  ✓ MLP intermediate activation sparsity")
    print("  ✓ Layer normalization statistics")
    print("  ✓ Residual stream decomposition")
    print("  ✓ Attention head behavioral classification")
    print("  ✓ Real-time numerical validation")
    print()
    print("=" * 80)


def main():
    print("=" * 80)
    print("EventChains LLM Explainability Demo")
    print("Capturing Data That Hugging Face Misses")
    print("=" * 80)
    print()

    # Ensure dependencies
    ensure_dependencies()

    # Download model
    model, tokenizer = download_model()

    # Create instrumented chain
    print("=" * 80)
    print("Building Instrumented Inference Chain")
    print("=" * 80)
    print("Creating EventChain with:")
    print("  - 6 transformer layer events")
    print("  - K/Q/V norm tracking middleware")
    print("  - MLP sparsity tracking middleware")
    print("  - Attention head analyzer middleware")
    print("  - Numerical validation middleware")
    print("  - Residual stream analyzer middleware")
    print()

    chain, middleware_dict = create_instrumented_chain(model)

    print(f"✓ Chain built: {chain}")
    print()

    # Run inference
    print("=" * 80)
    print("Running Inference with Full Instrumentation")
    print("=" * 80)

    input_text = "The quick brown fox"
    print(f"Input: \"{input_text}\"")
    print()
    print("Executing chain...")

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

    # Print results
    print_results(context, middleware_dict)

    print("\n" + "=" * 80)
    print("EventChains: Making LLMs Observable")
    print("=" * 80)


if __name__ == "__main__":
    main()
"""
Transformer Invertibility Validation using EventChains
Properly following the EventChains pattern where ALL logic is in events.
"""
import time
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from eventchains import EventChain, EventContext, ChainableEvent, Result
from eventchains_ml.events import CollisionDetectionEvent, TokenCandidateEvent, ForwardPassInversionEvent, \
    VerifyAcceptanceEvent
from eventchains_ml.middleware import CollisionLoggingMiddleware, InversionMetricsMiddleware, MarginMiddleware


# ============================================================================
# SETUP EVENTS
# ============================================================================

class InitializeModelEvent(ChainableEvent):
    """Initialize the GPT-2 model and tokenizer"""

    def __init__(self, model_name='gpt2'):
        self.model_name = model_name

    def execute(self, context):
        gpt_model = GPT2LMHeadModel.from_pretrained(self.model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        gpt_model.eval()

        # Wrap the model with forward_to_layer capability
        class ModelWrapper:
            def __init__(self, model):
                self.transformer = model.transformer
                self._model = model

            def forward_to_layer(self, tokens, layer_idx):
                """Get hidden states at a specific layer"""
                if isinstance(tokens, list):
                    tokens = torch.tensor([tokens])
                elif tokens.dim() == 1:
                    tokens = tokens.unsqueeze(0)

                with torch.no_grad():
                    outputs = self.transformer(
                        tokens,
                        output_hidden_states=True
                    )
                    # hidden_states is a tuple: (embedding_output, layer_0, layer_1, ..., layer_n)
                    hidden_state = outputs.hidden_states[layer_idx + 1]
                    # Return the last token's hidden state
                    return hidden_state[0, -1, :]

        wrapped_model = ModelWrapper(gpt_model)

        context.set('model', wrapped_model)
        context.set('tokenizer', tokenizer)
        context.set('vocabulary', list(range(tokenizer.vocab_size)))

        return Result.ok()


class ForwardToLayerEvent(ChainableEvent):
    """
    Get hidden states at a specific layer.

    Reads from context:
        - 'model': Wrapped model with forward_to_layer method
        - 'tokens': List of token IDs or tensor
        - 'layer_idx': Layer index to extract

    Sets in context:
        - 'hidden_state': Hidden state tensor at the specified layer
    """

    def execute(self, context):
        model = context.get('model')
        tokens = context.get('tokens')
        layer_idx = context.get('layer_idx')

        hidden_state = model.forward_to_layer(tokens, layer_idx)

        context.set('hidden_state', hidden_state)
        return Result.ok()


# ============================================================================
# TEST 1: COLLISION RATE EVENTS
# ============================================================================

class GenerateRandomPrefixEvent(ChainableEvent):
    """Generate a random prefix for collision testing"""

    def execute(self, context):
        prefix_length = torch.randint(1, 10, (1,)).item()
        prefix = torch.randint(0, 100, (prefix_length,)).tolist()
        context.set('prefix', prefix)
        return Result.ok()


class SetupCollisionTestEvent(ChainableEvent):
    """Set up context for collision detection"""

    def __init__(self, layer_idx=6, vocab_size=100):
        self.layer_idx = layer_idx
        self.vocab_size = vocab_size

    def execute(self, context):
        vocabulary = list(range(self.vocab_size))
        context.set('vocabulary', vocabulary)
        context.set('layer_idx', self.layer_idx)
        return Result.ok()


class AggregateCollisionResultsEvent(ChainableEvent):
    """Aggregate collision test results"""

    def execute(self, context):
        num_collisions = context.get('num_collisions', 0)

        # Get or initialize totals
        total_collisions = context.get('total_collisions', 0)
        samples_tested = context.get('samples_tested', 0)

        context.set('total_collisions', total_collisions + num_collisions)
        context.set('samples_tested', samples_tested + 1)

        return Result.ok()


class PrintCollisionResultsEvent(ChainableEvent):
    """Print final collision test results"""

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def execute(self, context):
        total_collisions = context.get('total_collisions', 0)
        collision_rate = total_collisions / (self.num_samples * 100)

        print(f"\nRESULTS:")
        print(f"Samples: {self.num_samples}")
        print(f"Total collisions: {total_collisions}")
        print(f"Collision rate: {collision_rate:.4f}")
        print(f"Thesis prediction: 0.0000")
        print(f"Match: {'✓ PASS' if collision_rate < 0.001 else '✗ FAIL'}")

        context.set('collision_rate', collision_rate)
        context.set('collision_pass', collision_rate < 0.001)

        return Result.ok()


# ============================================================================
# TEST 2: SIPIT INVERSION EVENTS
# ============================================================================

class GenerateInversionPromptEvent(ChainableEvent):
    """Generate a random prompt for inversion testing"""

    def execute(self, context):
        prompt = torch.randint(0, 100, (20,)).tolist()
        true_last_token = prompt[-1]

        context.set('prompt', prompt)
        context.set('true_last_token', true_last_token)
        context.set('prefix', prompt[:-1])

        return Result.ok()


class GetTargetHiddenStateEvent(ChainableEvent):
    """Get the target hidden state we're trying to invert to"""

    def __init__(self, layer_idx=6):
        self.layer_idx = layer_idx

    def execute(self, context):
        prompt = context.get('prompt')
        context.set('tokens', prompt)
        context.set('layer_idx', self.layer_idx)

        # Use ForwardToLayerEvent to get hidden state
        forward_event = ForwardToLayerEvent()
        result = forward_event.execute(context)

        if result.is_success():
            hidden_state = context.get('hidden_state')
            context.set('target_hidden', hidden_state)

        return result


class SetupInversionPolicyEvent(ChainableEvent):
    """Set up the uniform policy for token search"""

    def execute(self, context):
        class UniformPolicy:
            def __init__(self, vocab):
                self.vocab = vocab
                self.idx = 0

            def next_candidate(self):
                if self.idx >= len(self.vocab):
                    return None
                token = self.vocab[self.idx]
                self.idx += 1
                return token

        policy = UniformPolicy(list(range(100)))
        context.set('policy', policy)
        context.set('attempts', 0)
        context.set('verified', False)

        return Result.ok()


class RunInversionLoopEvent(ChainableEvent):
    """Run the inversion loop to find the token"""

    def __init__(self, max_attempts=100):
        self.max_attempts = max_attempts

        # Build the inversion sub-chain - model will come from context
        self.inversion_chain = (EventChain()
            .add_event(TokenCandidateEvent())
            .add_event(ForwardPassInversionEvent(layer_idx=6))  # Gets model from context
            .add_event(VerifyAcceptanceEvent(epsilon=1e-6))
            .use_middleware(MarginMiddleware()))

    def execute(self, context):
        start_time = time.time()

        # The model is already wrapped with forward_to_layer method
        # No need to wrap it again

        # Run the inversion loop
        for attempt in range(self.max_attempts):
            self.inversion_chain.execute(context)
            if context.get('verified'):
                break

        elapsed = time.time() - start_time
        context.set('inversion_time', elapsed)

        return Result.ok()


class ValidateInversionResultEvent(ChainableEvent):
    """Validate that the recovered token matches the true token"""

    def execute(self, context):
        verified = context.get('verified', False)

        if verified:
            recovered_token = context.get('recovered_token')
            true_token = context.get('true_last_token')

            if recovered_token == true_token:
                context.set('inversion_success', True)
            else:
                context.set('inversion_success', False)
                print(f"Warning: Verified but wrong token. Expected {true_token}, got {recovered_token}")
        else:
            context.set('inversion_success', False)

        return Result.ok()


class AggregateInversionResultsEvent(ChainableEvent):
    """Aggregate inversion test results"""

    def execute(self, context):
        success = context.get('inversion_success', False)

        # Get or initialize totals
        total_successes = context.get('total_inversion_successes', 0)
        total_attempts = context.get('total_inversion_attempts', 0)
        total_time = context.get('total_inversion_time', 0.0)
        prompts_tested = context.get('prompts_tested', 0)

        if success:
            total_successes += 1
            total_attempts += context.get('attempts', 0)
            total_time += context.get('inversion_time', 0.0)

        context.set('total_inversion_successes', total_successes)
        context.set('total_inversion_attempts', total_attempts)
        context.set('total_inversion_time', total_time)
        context.set('prompts_tested', prompts_tested + 1)

        return Result.ok()


class PrintInversionResultsEvent(ChainableEvent):
    """Print final inversion test results"""

    def __init__(self, test_prompts):
        self.test_prompts = test_prompts

    def execute(self, context):
        successes = context.get('total_inversion_successes', 0)
        total_attempts = context.get('total_inversion_attempts', 0)
        total_time = context.get('total_inversion_time', 0.0)

        accuracy = 100 * successes / self.test_prompts if self.test_prompts > 0 else 0
        avg_attempts = total_attempts / successes if successes > 0 else 0
        avg_time = total_time / successes if successes > 0 else 0

        print(f"\nRESULTS:")
        print(f"Test prompts: {self.test_prompts}")
        print(f"Successful inversions: {successes}")
        print(f"Accuracy: {accuracy:.1f}%")
        print(f"Avg attempts: {avg_attempts:.1f}")
        print(f"Avg time: {avg_time:.3f}s")
        print(f"Thesis prediction: 100% accuracy, O(vocab_size) time")
        print(f"Match: {'✓ PASS' if successes == self.test_prompts else '✗ FAIL'}")

        context.set('inversion_accuracy', accuracy / 100)
        context.set('inversion_pass', successes == self.test_prompts)

        return Result.ok()


# ============================================================================
# TEST 3: SEPARATION MARGIN EVENTS
# ============================================================================

class GenerateMarginTestPrefixEvent(ChainableEvent):
    """Generate a random prefix for margin testing"""

    def execute(self, context):
        prefix = torch.randint(0, 100, (10,)).tolist()
        context.set('prefix', prefix)
        return Result.ok()


class ComputePairwiseDistancesEvent(ChainableEvent):
    """Compute pairwise distances between token hidden states"""

    def __init__(self, layer_idx=6, num_tokens=10):
        self.layer_idx = layer_idx
        self.num_tokens = num_tokens

    def execute(self, context):
        model = context.get('model')
        prefix = context.get('prefix')

        # Compute hidden states for each token using the wrapped model
        hiddens = []
        for token in range(self.num_tokens):
            tokens = prefix + [token]
            hidden_state = model.forward_to_layer(tokens, self.layer_idx)
            hiddens.append(hidden_state)

        # Compute minimum pairwise distance
        min_dist = float('inf')
        for i in range(len(hiddens)):
            for j in range(i + 1, len(hiddens)):
                dist = torch.norm(hiddens[i] - hiddens[j]).item()
                min_dist = min(min_dist, dist)

        context.set('min_margin', min_dist)
        return Result.ok()


class AggregateMarginResultsEvent(ChainableEvent):
    """Aggregate margin test results"""

    def execute(self, context):
        margin = context.get('min_margin')

        # Get or initialize lists
        all_margins = context.get('all_margins', [])
        all_margins.append(margin)

        context.set('all_margins', all_margins)

        return Result.ok()


class PrintMarginResultsEvent(ChainableEvent):
    """Print final margin test results"""

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def execute(self, context):
        margins = context.get('all_margins', [])

        positive_margins = sum(1 for m in margins if m > 0)
        min_margin = min(margins) if margins else 0
        max_margin = max(margins) if margins else 0
        mean_margin = sum(margins) / len(margins) if margins else 0

        print(f"\nRESULTS:")
        print(f"Samples: {self.num_samples}")
        print(f"Positive margins: {positive_margins}")
        print(f"Rate: {100 * positive_margins / self.num_samples:.1f}%")
        print(f"Min margin: {min_margin:.6f}")
        print(f"Max margin: {max_margin:.6f}")
        print(f"Mean margin: {mean_margin:.6f}")
        print(f"Thesis prediction: 100% positive")
        print(f"Match: {'✓ PASS' if positive_margins == self.num_samples else '✗ FAIL'}")

        context.set('margin_rate', positive_margins / self.num_samples)
        context.set('margin_pass', positive_margins == self.num_samples)

        return Result.ok()


# ============================================================================
# ORCHESTRATION EVENTS
# ============================================================================

class PrintHeaderEvent(ChainableEvent):
    """Print a test header"""

    def __init__(self, title):
        self.title = title

    def execute(self, context):
        print("=" * 80)
        print(self.title)
        print("=" * 80)
        return Result.ok()


class LoopEvent(ChainableEvent):
    """Execute a chain multiple times"""

    def __init__(self, chain, num_iterations):
        self.chain = chain
        self.num_iterations = num_iterations

    def execute(self, context):
        for i in range(self.num_iterations):
            self.chain.execute(context)
        return Result.ok()


class PrintFinalSummaryEvent(ChainableEvent):
    """Print the final summary of all tests"""

    def execute(self, context):
        collision_rate = context.get('collision_rate', 0)
        inversion_accuracy = context.get('inversion_accuracy', 0)
        margin_rate = context.get('margin_rate', 0)

        collision_pass = context.get('collision_pass', False)
        inversion_pass = context.get('inversion_pass', False)
        margin_pass = context.get('margin_pass', False)

        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        print(f"Collision rate: {collision_rate:.4f} (expect: ~0)")
        print(f"Inversion accuracy: {100 * inversion_accuracy:.1f}% (expect: 100%)")
        print(f"Positive margins: {100 * margin_rate:.1f}% (expect: 100%)")
        print()

        if collision_pass and inversion_pass and margin_pass:
            print("✓ THESIS VALIDATED")
        else:
            print("✗ THESIS QUESTIONED - Further investigation needed")

        return Result.ok()


# ============================================================================
# MAIN TEST CHAIN BUILDER
# ============================================================================

def build_collision_test_chain(num_samples=10):
    """Build the collision rate test chain"""

    # Single iteration chain
    single_test = (EventChain()
        .add_event(GenerateRandomPrefixEvent())
        .add_event(SetupCollisionTestEvent())
        .add_event(CollisionDetectionEvent())
        .add_event(AggregateCollisionResultsEvent())
        .use_middleware(CollisionLoggingMiddleware()))

    # Full test chain
    return (EventChain()
        .add_event(PrintHeaderEvent("TEST 1: COLLISION RATE (Monte Carlo)"))
        .add_event(LoopEvent(single_test, num_samples))
        .add_event(PrintCollisionResultsEvent(num_samples)))


def build_inversion_test_chain(test_prompts=20):
    """Build the SIPIT inversion test chain"""

    # Single inversion test
    single_test = (EventChain()
        .add_event(GenerateInversionPromptEvent())
        .add_event(GetTargetHiddenStateEvent())
        .add_event(SetupInversionPolicyEvent())
        .add_event(RunInversionLoopEvent())
        .add_event(ValidateInversionResultEvent())
        .add_event(AggregateInversionResultsEvent()))

    # Full test chain
    return (EventChain()
        .add_event(PrintHeaderEvent("TEST 2: SIPIT INVERSION"))
        .add_event(LoopEvent(single_test, test_prompts))
        .add_event(PrintInversionResultsEvent(test_prompts)))


def build_margin_test_chain(num_samples=10):
    """Build the separation margin test chain"""

    # Single margin test
    single_test = (EventChain()
        .add_event(GenerateMarginTestPrefixEvent())
        .add_event(ComputePairwiseDistancesEvent())
        .add_event(AggregateMarginResultsEvent()))

    # Full test chain
    return (EventChain()
        .add_event(PrintHeaderEvent("TEST 3: SEPARATION MARGIN"))
        .add_event(LoopEvent(single_test, num_samples))
        .add_event(PrintMarginResultsEvent(num_samples)))


def build_full_validation_chain(collision_samples=10, inversion_prompts=20, margin_samples=10):
    """Build the complete validation test chain"""

    return (EventChain()
        .add_event(PrintHeaderEvent("TRANSFORMER INVERTIBILITY VALIDATION\nUsing EventChains to test arXiv:2510.15511v3\n"))
        .add_event(InitializeModelEvent())
        .add_event(LoopEvent(build_collision_test_chain(collision_samples), 1))
        .add_event(LoopEvent(build_inversion_test_chain(inversion_prompts), 1))
        .add_event(LoopEvent(build_margin_test_chain(margin_samples), 1))
        .add_event(PrintFinalSummaryEvent()))


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run the complete validation"""

    # Build the master chain
    validation_chain = build_full_validation_chain(
        collision_samples=1000,
        inversion_prompts=100,
        margin_samples=100
    )

    # Execute with empty context
    context = EventContext({})
    validation_chain.execute(context)


if __name__ == '__main__':
    main()
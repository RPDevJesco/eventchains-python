import time
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from eventchains import EventChain, EventContext
from eventchains_ml.events import CollisionDetectionEvent, TokenCandidateEvent, ForwardPassInversionEvent, \
    VerifyAcceptanceEvent
from eventchains_ml.middleware import CollisionLoggingMiddleware, InversionMetricsMiddleware, MarginMiddleware


class TransformerInvertibilityValidator:
    """
    Validates the thesis: "Transformers are Almost-Surely Injective"
    Using EventChains to systematically test each claim.
    """

    def __init__(self, model_name='gpt2'):
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.vocabulary = list(range(self.tokenizer.vocab_size))
        self.model.eval()  # Set to evaluation mode

    def forward_to_layer(self, tokens, layer_idx):
        """
        Get hidden states at a specific layer.

        Args:
            tokens: List of token IDs or tensor
            layer_idx: Layer index to extract (0-based)

        Returns:
            Hidden state tensor at the specified layer
        """
        if isinstance(tokens, list):
            tokens = torch.tensor([tokens])
        elif tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        with torch.no_grad():
            outputs = self.model.transformer(
                tokens,
                output_hidden_states=True
            )
            # hidden_states is a tuple: (embedding_output, layer_0, layer_1, ..., layer_n)
            # So layer_idx+1 because index 0 is the embedding
            hidden_state = outputs.hidden_states[layer_idx + 1]
            # Return the last token's hidden state
            return hidden_state[0, -1, :]

    def test_collision_rate(self, num_samples=1000, layer_idx=6):
        """
        Test Theorem 2.2: Almost-sure injectivity at initialization

        Thesis prediction: Pr[collision] = 0
        """
        print("=" * 80)
        print("TEST 1: COLLISION RATE (Monte Carlo)")
        print("=" * 80)

        collision_chain = (EventChain()
                           .add_event(CollisionDetectionEvent())
                           .use_middleware(CollisionLoggingMiddleware()))

        total_collisions = 0

        for sample in range(num_samples):
            # Random prefix
            prefix_length = torch.randint(1, 10, (1,)).item()
            prefix = torch.randint(0, 100, (prefix_length,)).tolist()

            context = EventContext({
                'model': self,  # Pass self so events can use forward_to_layer
                'vocabulary': self.vocabulary[:100],  # Subset for speed
                'layer_idx': layer_idx,
                'prefix': prefix
            })

            collision_chain.execute(context)

            total_collisions += context.get('num_collisions', 0)

        collision_rate = total_collisions / (num_samples * 100)

        print(f"\nRESULTS:")
        print(f"Samples: {num_samples}")
        print(f"Total collisions: {total_collisions}")
        print(f"Collision rate: {collision_rate:.4f}")
        print(f"Thesis prediction: 0.0000")
        print(f"Match: {'✓ PASS' if collision_rate < 0.001 else '✗ FAIL'}")

        return collision_rate

    def test_sipit_inversion(self, test_prompts=100):
        """
        Test Algorithm 1 (SIPIT): Sequential Inversion via Policy and Iteration

        Thesis prediction: Exact recovery in linear time
        """
        print("\n" + "=" * 80)
        print("TEST 2: SIPIT INVERSION")
        print("=" * 80)

        # Build inversion chain
        metrics = InversionMetricsMiddleware()

        inversion_chain = (EventChain()
                           .add_event(TokenCandidateEvent(policy=None))  # Set per execution
                           .add_event(ForwardPassInversionEvent(self, layer_idx=6))
                           .add_event(VerifyAcceptanceEvent(epsilon=0.0))
                           .use_middleware(MarginMiddleware())
                           .use_middleware(metrics))

        successes = 0
        total_attempts = 0
        times = []

        for i in range(test_prompts):
            # Generate random prompt
            prompt = torch.randint(0, 100, (20,)).tolist()

            # Get target hidden state
            with torch.no_grad():
                target_hidden = self.forward_to_layer(prompt, 6)

            # Try to recover last token
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

            context = EventContext({
                'target_hidden': target_hidden,
                'prefix': prompt[:-1],
                'model': self,
                'policy': UniformPolicy(list(range(100))),
                'attempts': 0
            })

            # Run inversion
            start = time.time()
            for _ in range(100):  # Max attempts
                inversion_chain.execute(context)
                if context.get('verified'):
                    break
            elapsed = time.time() - start

            if context.get('verified'):
                successes += 1
                times.append(elapsed)
                total_attempts += context.get('attempts', 0)

        print(f"\nRESULTS:")
        print(f"Test prompts: {test_prompts}")
        print(f"Successful inversions: {successes}")
        print(f"Accuracy: {100 * successes / test_prompts:.1f}%")
        print(f"Avg attempts: {total_attempts / successes if successes > 0 else 0:.1f}")
        print(f"Avg time: {sum(times) / len(times) if times else 0:.3f}s")
        print(f"Thesis prediction: 100% accuracy, O(vocab_size) time")
        print(f"Match: {'✓ PASS' if successes == test_prompts else '✗ FAIL'}")

        return successes / test_prompts

    def test_separation_margin(self, num_samples=100):
        """
        Test Lemma D.1: Strict separation margin almost surely

        Thesis prediction: Δ > 0 with probability 1
        """
        print("\n" + "=" * 80)
        print("TEST 3: SEPARATION MARGIN")
        print("=" * 80)

        margins = []

        for sample in range(num_samples):
            # Random prefix
            prefix = torch.randint(0, 100, (10,)).tolist()

            # Compute pairwise distances
            min_dist = float('inf')

            with torch.no_grad():
                hiddens = []
                for token in range(10):  # Small vocab for speed
                    hidden = self.forward_to_layer(
                        prefix + [token], 6
                    )
                    hiddens.append(hidden)

                for i in range(len(hiddens)):
                    for j in range(i + 1, len(hiddens)):
                        dist = torch.norm(hiddens[i] - hiddens[j]).item()
                        min_dist = min(min_dist, dist)

            margins.append(min_dist)

        positive_margins = sum(1 for m in margins if m > 0)

        print(f"\nRESULTS:")
        print(f"Samples: {num_samples}")
        print(f"Positive margins: {positive_margins}")
        print(f"Rate: {100 * positive_margins / num_samples:.1f}%")
        print(f"Min margin: {min(margins):.6f}")
        print(f"Max margin: {max(margins):.6f}")
        print(f"Mean margin: {sum(margins) / len(margins):.6f}")
        print(f"Thesis prediction: 100% positive")
        print(f"Match: {'✓ PASS' if positive_margins == num_samples else '✗ FAIL'}")

        return positive_margins / num_samples


def main():
    """Run all tests"""
    validator = TransformerInvertibilityValidator(model_name='gpt2')

    print("TRANSFORMER INVERTIBILITY VALIDATION")
    print("Using EventChains to test arXiv:2510.15511v3")
    print()

    # Test 1: Collision rate
    collision_rate = validator.test_collision_rate(num_samples=100)

    # Test 2: SIPIT inversion
    accuracy = validator.test_sipit_inversion(test_prompts=20)

    # Test 3: Separation margin
    margin_rate = validator.test_separation_margin(num_samples=100)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Collision rate: {collision_rate:.4f} (expect: ~0)")
    print(f"Inversion accuracy: {100 * accuracy:.1f}% (expect: 100%)")
    print(f"Positive margins: {100 * margin_rate:.1f}% (expect: 100%)")
    print()

    if collision_rate < 0.001 and accuracy > 0.95 and margin_rate > 0.95:
        print("✓ THESIS VALIDATED")
    else:
        print("✗ THESIS QUESTIONED - Further investigation needed")


if __name__ == '__main__':
    main()
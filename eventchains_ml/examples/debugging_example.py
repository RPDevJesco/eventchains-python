"""
Debugging Example - Detecting Common Training Issues

This example demonstrates how EventChains ML helps detect and diagnose
common neural network training issues like vanishing gradients and dead neurons.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from eventchains import EventChain, EventContext
from eventchains_ml import (
    ForwardPassEvent,
    CalculateLossEvent,
    BackpropagationEvent,
    UpdateWeightsEvent,
    GradientMonitorMiddleware,
    DeadNeuronDetectorMiddleware,
)


class ProblemNet(nn.Module):
    """
    A deliberately problematic network to demonstrate issue detection.
    
    Issues:
    1. Very deep without normalization (vanishing gradients)
    2. Poor initialization
    3. ReLU neurons likely to die
    """
    
    def __init__(self, use_good_practices=False):
        super().__init__()
        
        if use_good_practices:
            # Good practices: batch norm, better initialization
            self.layers = nn.Sequential(
                nn.Linear(784, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
        else:
            # Problematic: no batch norm, deep network, will have issues
            self.layers = nn.Sequential(
                nn.Linear(784, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
            
            # Poor initialization - makes problems worse
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=0.01)  # Too small!
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = x.view(-1, 784)
        return self.layers(x)


def create_diagnostic_chain(criterion, optimizer):
    """Create a chain with diagnostic middleware."""
    gradient_monitor = GradientMonitorMiddleware(
        vanishing_threshold=1e-7,
        exploding_threshold=1e3,
        verbose=True
    )
    
    dead_neuron_detector = DeadNeuronDetectorMiddleware(
        threshold_percent=30.0,
        verbose=True
    )
    
    chain = (EventChain()
        .add_event(ForwardPassEvent(track_activations=True))
        .add_event(CalculateLossEvent(criterion))
        .add_event(BackpropagationEvent(track_gradients=True))
        .add_event(UpdateWeightsEvent(optimizer))
        .use_middleware(gradient_monitor)
        .use_middleware(dead_neuron_detector))
    
    return chain, gradient_monitor, dead_neuron_detector


def generate_dummy_batch(batch_size=32, device='cpu'):
    """Generate dummy data for testing."""
    batch = torch.randn(batch_size, 1, 28, 28).to(device)
    labels = torch.randint(0, 10, (batch_size,)).to(device)
    return batch, labels


def diagnose_model(model_name, model, device='cpu', num_batches=5):
    """
    Run a few training iterations to diagnose potential issues.
    """
    print(f"\n{'=' * 80}")
    print(f"Diagnosing: {model_name}")
    print('=' * 80)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    chain, gradient_monitor, dead_neuron_detector = create_diagnostic_chain(
        criterion, optimizer
    )
    
    model.train()
    
    issues_found = []
    
    for batch_idx in range(num_batches):
        print(f"\nBatch {batch_idx + 1}/{num_batches}")
        print("-" * 80)
        
        # Generate dummy batch
        batch, labels = generate_dummy_batch(device=device)
        
        # Create context
        context = EventContext({
            'model': model,
            'batch': batch,
            'labels': labels,
            'device': device,
        })
        
        # Execute training step
        result = chain.execute(context)
        
        if not result.success:
            print(f"✗ Training failed: {result.error}")
            issues_found.append(f"Training failure: {result.error}")
            break
        
        # Check for gradient issues
        if context.has('gradient_issues'):
            gradient_issues = context.get('gradient_issues')
            issues_found.extend(gradient_issues)
        
        # Check for dead neurons
        if context.has('dead_neurons'):
            dead_neurons = context.get('dead_neurons')
            issues_found.extend([
                f"Dead neurons in {info['layer']}: {info['zero_percent']:.1f}%"
                for info in dead_neurons
            ])
        
        # Print loss
        loss = context.get('loss_value', 0)
        print(f"Loss: {loss:.4f}")
    
    # Summary
    print(f"\n{'=' * 80}")
    print(f"Diagnosis Summary for {model_name}")
    print('=' * 80)
    
    if issues_found:
        print(f"\n❌ {len(issues_found)} issue(s) detected:\n")
        for i, issue in enumerate(issues_found[:10], 1):  # Show first 10
            print(f"  {i}. {issue}")
        if len(issues_found) > 10:
            print(f"  ... and {len(issues_found) - 10} more")
    else:
        print("\n✅ No major issues detected!")
        print("   Model appears to be training well.")
    
    return len(issues_found) == 0


def main():
    print("=" * 80)
    print("EventChains ML - Debugging Example")
    print("Detecting Common Training Issues")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}\n")
    
    # Test 1: Problematic network
    print("\n" + "🔬" * 40)
    print("TEST 1: Problematic Network")
    print("🔬" * 40)
    print("\nThis network has several issues:")
    print("  • Very deep without batch normalization")
    print("  • Poor weight initialization (too small)")
    print("  • Likely to have vanishing gradients")
    print("  • ReLU neurons likely to die")
    
    problematic_model = ProblemNet(use_good_practices=False).to(device)
    problem_free = diagnose_model("Problematic Network", problematic_model, device)
    
    # Test 2: Fixed network
    print("\n\n" + "🔬" * 40)
    print("TEST 2: Fixed Network (Good Practices)")
    print("🔬" * 40)
    print("\nThis network uses good practices:")
    print("  • Batch normalization between layers")
    print("  • Proper weight initialization")
    print("  • Better gradient flow")
    print("  • Healthier neuron activations")
    
    fixed_model = ProblemNet(use_good_practices=True).to(device)
    fixed_free = diagnose_model("Fixed Network", fixed_model, device)
    
    # Final comparison
    print("\n" + "=" * 80)
    print("COMPARISON & LEARNINGS")
    print("=" * 80)
    
    print("\n📊 Results:")
    print(f"  Problematic Network: {'✅ Passed' if problem_free else '❌ Issues detected'}")
    print(f"  Fixed Network:       {'✅ Passed' if fixed_free else '❌ Issues detected'}")
    
    print("\n💡 Key Takeaways:")
    print("  1. EventChains ML detected issues in minutes, not hours")
    print("  2. Gradient monitoring caught vanishing gradients immediately")
    print("  3. Dead neuron detection identified problematic layers")
    print("  4. Quick iteration: diagnose → fix → validate")
    print("  5. Complete visibility into what's happening during training")
    
    print("\n🎯 Development Workflow:")
    print("  1. Build network architecture")
    print("  2. Run diagnostic chain (5-10 batches)")
    print("  3. Fix detected issues")
    print("  4. Validate fixes with diagnostic chain")
    print("  5. Once validated, train full model confidently")
    
    print("\n" + "=" * 80)
    print("EventChains ML: Making Neural Networks Debuggable!")
    print("=" * 80)


if __name__ == "__main__":
    main()

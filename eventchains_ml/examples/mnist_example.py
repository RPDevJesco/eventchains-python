"""
MNIST Training Example with EventChains ML

This example demonstrates how EventChains ML brings complete observability
to neural network training, making it easy to detect and debug issues.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from datetime import datetime

from eventchains import EventChain, EventContext
from eventchains_ml import (
    LoadBatchEvent,
    ForwardPassEvent,
    CalculateLossEvent,
    BackpropagationEvent,
    UpdateWeightsEvent,
    ValidationEvent,
    GradientMonitorMiddleware,
    DeadNeuronDetectorMiddleware,
    PerformanceProfilerMiddleware,
)


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST classification."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        # Conv layers
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # FC layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def load_mnist_data(batch_size=64):
    """Load MNIST train and validation datasets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Training data
    train_dataset = datasets.MNIST(
        'data',
        train=True,
        download=True,
        transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Validation data
    val_dataset = datasets.MNIST(
        'data',
        train=False,
        download=True,
        transform=transform
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader


def create_training_chain(train_loader, criterion, optimizer, profiler):
    """Create the EventChain for training."""
    return (EventChain()
        .add_event(LoadBatchEvent(train_loader))
        .add_event(ForwardPassEvent(track_activations=True))
        .add_event(CalculateLossEvent(criterion))
        .add_event(BackpropagationEvent(track_gradients=True))
        .add_event(UpdateWeightsEvent(optimizer))
        .use_middleware(GradientMonitorMiddleware(
            vanishing_threshold=1e-7,
            exploding_threshold=1e3,
            verbose=False  # Set to True to see gradient warnings
        ))
        .use_middleware(DeadNeuronDetectorMiddleware(
            threshold_percent=30.0,
            verbose=False  # Set to True to see dead neuron warnings
        ))
        .use_middleware(profiler))


def create_validation_chain(val_loader, criterion):
    """Create the EventChain for validation."""
    return (EventChain()
        .add_event(ValidationEvent(val_loader, compute_accuracy=True)))


def train_epoch(chain, model, device, epoch, num_batches, print_every=50):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    batches_processed = 0
    
    for batch_idx in range(num_batches):
        context = EventContext({
            'model': model,
            'device': device,
            'epoch': epoch,
            'batch_idx': batch_idx,
        })
        
        result = chain.execute(context)
        
        if not result.success:
            print(f"✗ Training failed at batch {batch_idx}: {result.error}")
            break
        
        loss = context.get('loss_value', 0)
        total_loss += loss
        batches_processed += 1
        
        # Print progress
        if (batch_idx + 1) % print_every == 0:
            avg_loss = total_loss / batches_processed
            print(f"  Batch [{batch_idx + 1}/{num_batches}] "
                  f"Loss: {loss:.4f} (Avg: {avg_loss:.4f})")
    
    avg_loss = total_loss / batches_processed if batches_processed > 0 else 0
    return avg_loss


def validate(chain, model, device, criterion):
    """Run validation."""
    context = EventContext({
        'model': model,
        'device': device,
        'criterion': criterion,
    })

    result = chain.execute(context)

    if result.success:
        val_loss = context.get('val_loss', 0)
        val_accuracy = context.get('val_accuracy', 0)
        return val_loss, val_accuracy
    else:
        print(f"✗ Validation failed: {result.error}")
        return None, None


def main():
    now = datetime.now()
    # Format the time as a string (e.g., HH:MM:SS)
    current_time = now.strftime("%H:%M:%S")

    # Print the formatted current time
    print("Current Time =", current_time)
    print("=" * 80)
    print("MNIST Training with EventChains ML")
    print("Full Observability & Debugging Example")
    print("=" * 80)
    print()

    # Configuration
    batch_size = 64
    num_epochs = 3
    batches_per_epoch = 200  # Train on subset for demo
    learning_rate = 0.001

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print()

    # Load data
    print("Loading MNIST data...")
    train_loader, val_loader = load_mnist_data(batch_size)
    print(f"✓ Training batches: {len(train_loader)}")
    print(f"✓ Validation batches: {len(val_loader)}")
    print()

    # Create model
    print("Creating model...")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Create performance profiler
    profiler = PerformanceProfilerMiddleware()

    # Create chains
    training_chain = create_training_chain(train_loader, criterion, optimizer, profiler)
    validation_chain = create_validation_chain(val_loader, criterion)

    print(f"✓ Training chain: {training_chain}")
    print()

    # Training loop
    print("=" * 80)
    print("Starting Training")
    print("=" * 80)
    print()

    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print('=' * 80)

        # Train
        avg_loss = train_epoch(
            training_chain,
            model,
            device,
            epoch,
            batches_per_epoch,
            print_every=50
        )

        print(f"\n  Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

        # Validate
        print("\n  Running validation...")
        val_loss, val_accuracy = validate(validation_chain, model, device, criterion)

        if val_accuracy is not None:
            print(f"  Validation Loss: {val_loss:.4f}")
            print(f"  Validation Accuracy: {val_accuracy:.2f}%")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                print(f"  ✓ New best accuracy!")

    # Final results
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nBest Validation Accuracy: {best_val_accuracy:.2f}%")

    # Performance report
    print("\n" + "=" * 80)
    print("Performance Analysis")
    print("=" * 80)
    profiler.print_report()

    print("\n" + "=" * 80)
    print("Key Benefits of EventChains ML:")
    print("=" * 80)
    print("✓ Complete visibility into training process")
    print("✓ Real-time gradient monitoring")
    print("✓ Dead neuron detection")
    print("✓ Performance profiling per event")
    print("✓ Easy debugging when issues occur")
    print("✓ Full audit trail for reproducibility")
    print("=" * 80)
    print("Current Time =", current_time)

if __name__ == "__main__":
    main()
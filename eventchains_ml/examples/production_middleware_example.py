"""
Production Middleware Example - Comprehensive ML Training with All Middleware

This example demonstrates all production-ready middleware working together:
1. AuditLogMiddleware - Complete experiment tracking and reproducibility
2. ValidationMiddleware - Real-time data validation and error detection
3. MetricsCollectorMiddleware - Prometheus/monitoring integration
4. CompressionMiddleware - Memory-efficient storage of large tensors

Shows how EventChains ML enables production-grade ML workflows with:
- Full audit trails for compliance
- Automatic data validation
- Monitoring system integration
- Memory optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from eventchains import EventChain, EventContext
from eventchains_ml import (
    LoadBatchEvent,
    ForwardPassEvent,
    CalculateLossEvent,
    BackpropagationEvent,
    UpdateWeightsEvent,
    ValidationEvent,
    AuditLogMiddleware,
    ValidationMiddleware,
    MetricsCollectorMiddleware,
    CompressionMiddleware,
    PerformanceProfilerMiddleware,
)


class ProductionCNN(nn.Module):
    """Production-ready CNN for MNIST classification."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = x.view(-1, 64 * 7 * 7)
        
        x = self.relu(self.bn3(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def load_mnist_data(batch_size=64):
    """Load MNIST train and validation datasets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
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


def create_production_training_chain(
    train_loader,
    criterion,
    optimizer,
    audit_log,
    validation_mw,
    metrics_collector,
    compression_mw,
    profiler
):
    """
    Create a production-ready training chain with all middleware.
    
    Middleware stack (LIFO order):
    - CompressionMiddleware (outermost - compresses large tensors)
    - MetricsCollectorMiddleware (collects metrics for monitoring)
    - ValidationMiddleware (validates data integrity)
    - AuditLogMiddleware (logs everything for audit trail)
    - PerformanceProfilerMiddleware (innermost - times events)
    """
    return (EventChain()
        .add_event(LoadBatchEvent(train_loader))
        .add_event(ForwardPassEvent(track_activations=True))
        .add_event(CalculateLossEvent(criterion))
        .add_event(BackpropagationEvent(track_gradients=True))
        .add_event(UpdateWeightsEvent(optimizer))
        .use_middleware(profiler)
        .use_middleware(audit_log)
        .use_middleware(validation_mw)
        .use_middleware(metrics_collector)
        .use_middleware(compression_mw))


def train_epoch(chain, model, device, epoch, num_batches, print_every=50):
    """Train for one epoch with full production monitoring."""
    model.train()
    
    total_loss = 0.0
    batches_processed = 0
    
    for batch_idx in range(num_batches):
        context = EventContext({
            'model': model,
            'device': device,
            'epoch': epoch,
            'batch_idx': batch_idx,
            'experiment_id': 'mnist_production_v1',
            'hyperparameters': {
                'learning_rate': 0.001,
                'batch_size': 64,
                'optimizer': 'Adam',
            }
        })
        
        result = chain.execute(context)
        
        if not result.success:
            print(f"✗ Training failed at batch {batch_idx}: {result.error}")
            break
        
        loss = context.get('loss_value', 0)
        total_loss += loss
        batches_processed += 1
        
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
    print("=" * 80)
    print("Production Middleware Example")
    print("EventChains ML with Full Production Stack")
    print("=" * 80)
    print()

    batch_size = 64
    num_epochs = 2
    batches_per_epoch = 100
    learning_rate = 0.001

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print()

    print("Loading MNIST data...")
    train_loader, val_loader = load_mnist_data(batch_size)
    print(f"✓ Training batches: {len(train_loader)}")
    print(f"✓ Validation batches: {len(val_loader)}")
    print()

    print("Creating model...")
    model = ProductionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    print("=" * 80)
    print("Initializing Production Middleware Stack")
    print("=" * 80)
    print()

    print("1. AuditLogMiddleware")
    print("   - Logs every event execution to audit_log.jsonl")
    print("   - Tracks hyperparameters, metrics, and decisions")
    print("   - Enables full experiment reproducibility")
    audit_log = AuditLogMiddleware(
        log_file='production_audit_log.jsonl',
        log_to_console=False
    )
    print(f"   ✓ Session ID: {audit_log.session_id}")
    print()

    print("2. ValidationMiddleware")
    print("   - Validates tensor shapes and data ranges")
    print("   - Detects NaN/Inf values in real-time")
    print("   - Prevents training on corrupted data")
    validation_mw = ValidationMiddleware(
        strict=True,
        verbose=False
    )
    print("   ✓ Strict mode enabled (fails on validation errors)")
    print()

    print("3. MetricsCollectorMiddleware")
    print("   - Collects metrics for Prometheus/monitoring")
    print("   - Tracks event counts, durations, failures")
    print("   - Exports in multiple formats (Prometheus, JSON, StatsD)")
    metrics_collector = MetricsCollectorMiddleware(
        export_format='prometheus',
        export_file='production_metrics.txt'
    )
    print("   ✓ Prometheus format enabled")
    print()

    print("4. CompressionMiddleware")
    print("   - Compresses large tensors in context")
    print("   - Reduces memory usage for activation storage")
    print("   - Configurable compression level and threshold")
    compression_mw = CompressionMiddleware(
        compress_keys=['layer_activations'],
        compression_level=6,
        threshold_mb=0.5
    )
    print("   ✓ Compressing activations >0.5 MB")
    print()

    print("5. PerformanceProfilerMiddleware")
    print("   - Times each event execution")
    print("   - Identifies performance bottlenecks")
    profiler = PerformanceProfilerMiddleware()
    print("   ✓ Profiling enabled")
    print()

    # Create chains
    training_chain = create_production_training_chain(
        train_loader,
        criterion,
        optimizer,
        audit_log,
        validation_mw,
        metrics_collector,
        compression_mw,
        profiler
    )
    
    validation_chain = (EventChain()
        .add_event(ValidationEvent(val_loader, compute_accuracy=True)))

    print(f"✓ Training chain: {training_chain}")
    print()

    print("=" * 80)
    print("Starting Production Training")
    print("=" * 80)
    print()

    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print('=' * 80)

        avg_loss = train_epoch(
            training_chain,
            model,
            device,
            epoch,
            batches_per_epoch,
            print_every=25
        )

        print(f"\n  Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

        print("\n  Running validation...")
        val_loss, val_accuracy = validate(validation_chain, model, device, criterion)

        if val_accuracy is not None:
            print(f"  Validation Loss: {val_loss:.4f}")
            print(f"  Validation Accuracy: {val_accuracy:.2f}%")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                print(f"  ✓ New best accuracy!")

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nBest Validation Accuracy: {best_val_accuracy:.2f}%")

    print("\n" + "=" * 80)
    print("Production Middleware Reports")
    print("=" * 80)

    print("\n1. PERFORMANCE PROFILING")
    profiler.print_report()

    print("\n2. METRICS COLLECTION")
    print("=" * 80)
    metrics_summary = metrics_collector.get_summary()
    print(f"Total Events Executed: {metrics_summary['total_events']}")
    print(f"Total Failures: {metrics_summary['total_failures']}")
    print("\nPer-Event Metrics:")
    for event, stats in metrics_summary['events'].items():
        print(f"  {event}:")
        print(f"    Count: {stats['count']}")
        print(f"    Failures: {stats['failures']}")
        print(f"    Avg Duration: {stats['avg_duration_ms']:.2f}ms")
    
    print("\nExporting metrics to production_metrics.txt...")
    metrics_collector.export_metrics()
    print("✓ Metrics exported in Prometheus format")

    print("\n3. COMPRESSION STATISTICS")
    compression_mw.print_stats()

    print("\n4. VALIDATION REPORT")
    print("=" * 80)
    validation_errors = validation_mw.get_errors()
    if validation_errors:
        print(f"⚠️  {len(validation_errors)} validation error(s) detected:")
        for error in validation_errors[:10]:
            print(f"  - {error}")
    else:
        print("✅ No validation errors detected")
        print("   All data passed integrity checks")

    print("\n5. AUDIT LOG")
    print("=" * 80)
    print(f"Session ID: {audit_log.session_id}")
    print(f"Total Events Logged: {audit_log.event_counter}")
    print(f"Audit Log File: production_audit_log.jsonl")
    audit_log.close()
    print("✓ Audit log closed")

    print("\n" + "=" * 80)
    print("Production Benefits Demonstrated")
    print("=" * 80)
    print()
    print("✓ AUDIT TRAIL")
    print("  - Every training decision logged to JSONL")
    print("  - Full experiment reproducibility")
    print("  - Compliance-ready audit logs")
    print()
    print("✓ DATA VALIDATION")
    print("  - Real-time NaN/Inf detection")
    print("  - Shape mismatch prevention")
    print("  - Training on clean data guaranteed")
    print()
    print("✓ MONITORING INTEGRATION")
    print("  - Prometheus-compatible metrics")
    print("  - Event counts, durations, failures tracked")
    print("  - Ready for Grafana dashboards")
    print()
    print("✓ MEMORY OPTIMIZATION")
    print("  - Large tensors automatically compressed")
    print("  - Significant memory savings")
    print("  - Configurable compression strategy")
    print()
    print("✓ PERFORMANCE PROFILING")
    print("  - Per-event timing analysis")
    print("  - Bottleneck identification")
    print("  - Optimization guidance")
    print()
    print("=" * 80)
    print("EventChains ML: Production-Ready ML Training!")
    print("=" * 80)


if __name__ == "__main__":
    main()

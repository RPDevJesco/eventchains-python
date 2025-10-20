"""
Machine Learning Middleware for EventChains

Middleware that wraps ML training events to provide:
- Gradient monitoring and anomaly detection
- Dead neuron detection
- Performance profiling
- TensorBoard logging
- Early stopping
"""

import time
from eventchains import Middleware


class GradientMonitorMiddleware(Middleware):
    """
    Monitor gradients for common issues like vanishing or exploding gradients.
    
    Detects:
    - Vanishing gradients (norm < threshold)
    - Exploding gradients (norm > threshold)
    - NaN or Inf gradients
    """
    
    def __init__(self, vanishing_threshold=1e-7, exploding_threshold=1e3, verbose=True):
        """
        Initialize the GradientMonitorMiddleware.
        
        Args:
            vanishing_threshold: Threshold below which gradients are considered vanishing
            exploding_threshold: Threshold above which gradients are considered exploding
            verbose: Whether to print warnings
        """
        self.vanishing_threshold = vanishing_threshold
        self.exploding_threshold = exploding_threshold
        self.verbose = verbose
        self.issues_detected = []
    
    def execute(self, context, next_callable):
        result = next_callable(context)
        
        # Only check gradients after backpropagation
        event_name = context.get('_current_event', '')
        if 'Backpropagation' not in event_name:
            return result
        
        gradient_norms = context.get('gradient_norms', {})
        
        for layer, norm in gradient_norms.items():
            # Check for NaN or Inf
            if not isinstance(norm, (int, float)) or norm != norm or norm == float('inf'):
                issue = f"⚠️  NaN/Inf gradient detected in {layer}"
                self.issues_detected.append(issue)
                if self.verbose:
                    print(issue)
                continue
            
            # Check for vanishing gradients
            if norm < self.vanishing_threshold:
                issue = f"⚠️  Vanishing gradient in {layer}: {norm:.2e}"
                self.issues_detected.append(issue)
                if self.verbose:
                    print(issue)
            
            # Check for exploding gradients
            elif norm > self.exploding_threshold:
                issue = f"⚠️  Exploding gradient in {layer}: {norm:.2e}"
                self.issues_detected.append(issue)
                if self.verbose:
                    print(issue)
        
        # Store issues in context for later analysis
        if self.issues_detected:
            context.set('gradient_issues', self.issues_detected.copy())
        
        return result
    
    def get_issues(self):
        """Return all detected gradient issues."""
        return self.issues_detected.copy()
    
    def clear_issues(self):
        """Clear the list of detected issues."""
        self.issues_detected.clear()


class DeadNeuronDetectorMiddleware(Middleware):
    """
    Detect dead or dying neurons (ReLU neurons that always output zero).
    
    Monitors activation statistics to identify neurons that are consistently
    producing zero outputs, which indicates they're not contributing to learning.
    """
    
    def __init__(self, threshold_percent=30.0, verbose=True):
        """
        Initialize the DeadNeuronDetectorMiddleware.
        
        Args:
            threshold_percent: Percentage of zeros above which layer is flagged
            verbose: Whether to print warnings
        """
        self.threshold_percent = threshold_percent
        self.verbose = verbose
        self.dead_layers = []
    
    def execute(self, context, next_callable):
        result = next_callable(context)
        
        # Only check activations after forward pass
        event_name = context.get('_current_event', '')
        if 'ForwardPass' not in event_name:
            return result
        
        layer_activations = context.get('layer_activations', {})
        
        for layer, stats in layer_activations.items():
            zero_percent = stats.get('zero_percent', 0.0)
            
            if zero_percent > self.threshold_percent:
                dead_info = {
                    'layer': layer,
                    'zero_percent': zero_percent,
                    'zeros': stats.get('zeros', 0),
                    'total': stats.get('total', 0)
                }
                self.dead_layers.append(dead_info)
                
                if self.verbose:
                    print(f"⚠️  {layer}: {zero_percent:.1f}% dead neurons "
                          f"({stats.get('zeros', 0)}/{stats.get('total', 0)})")
        
        # Store dead neuron info in context
        if self.dead_layers:
            context.set('dead_neurons', self.dead_layers.copy())
        
        return result
    
    def get_dead_layers(self):
        """Return all layers with dead neurons."""
        return self.dead_layers.copy()
    
    def clear(self):
        """Clear the list of dead layers."""
        self.dead_layers.clear()


class PerformanceProfilerMiddleware(Middleware):
    """
    Profile the execution time and throughput of each event.
    
    Tracks:
    - Time per event (min, max, avg)
    - Samples per second
    - Total time spent in each event
    """
    
    def __init__(self):
        self.timings = {}
        self.call_counts = {}
        self.start_time = None
    
    def execute(self, context, next_callable):
        event_name = context.get('_current_event', 'Unknown')
        
        # Track overall timing
        if self.start_time is None:
            self.start_time = time.perf_counter()
        
        # Time the event execution
        start = time.perf_counter()
        result = next_callable(context)
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        
        # Store timing data
        if event_name not in self.timings:
            self.timings[event_name] = []
            self.call_counts[event_name] = 0
        
        self.timings[event_name].append(elapsed)
        self.call_counts[event_name] += 1
        
        # Add timing to context
        context.set(f'{event_name}_time_ms', elapsed)
        
        return result
    
    def get_report(self):
        """Generate a performance report."""
        report = []
        
        for event, times in sorted(self.timings.items()):
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            total_time = sum(times)
            calls = self.call_counts[event]
            
            report.append({
                'event': event,
                'avg_ms': avg_time,
                'min_ms': min_time,
                'max_ms': max_time,
                'total_ms': total_time,
                'calls': calls,
            })
        
        return report
    
    def print_report(self):
        """Print a formatted performance report."""
        print("\n" + "=" * 80)
        print("Performance Profiling Report")
        print("=" * 80)
        
        report = self.get_report()
        
        # Header
        print(f"{'Event':<30} {'Avg (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10} "
              f"{'Total (ms)':>12} {'Calls':>8}")
        print("-" * 80)
        
        # Data
        total_time_all = 0
        for entry in report:
            print(f"{entry['event']:<30} "
                  f"{entry['avg_ms']:>10.2f} "
                  f"{entry['min_ms']:>10.2f} "
                  f"{entry['max_ms']:>10.2f} "
                  f"{entry['total_ms']:>12.2f} "
                  f"{entry['calls']:>8}")
            total_time_all += entry['total_ms']
        
        print("-" * 80)
        print(f"{'TOTAL':<30} {'':<10} {'':<10} {'':<10} {total_time_all:>12.2f}")
        
        # Overall metrics
        if self.start_time:
            elapsed_seconds = time.perf_counter() - self.start_time
            print(f"\nTotal wall time: {elapsed_seconds:.2f}s")
            print(f"Overhead: {(total_time_all / 1000) / elapsed_seconds * 100:.1f}%")
        
        print("=" * 80)
    
    def reset(self):
        """Reset all timing data."""
        self.timings.clear()
        self.call_counts.clear()
        self.start_time = None


class TensorBoardMiddleware(Middleware):
    """
    Log training metrics to TensorBoard.
    
    Logs:
    - Loss values
    - Learning rate
    - Gradient norms
    - Activation statistics
    """
    
    def __init__(self, log_dir='runs', enabled=True):
        """
        Initialize the TensorBoardMiddleware.
        
        Args:
            log_dir: Directory for TensorBoard logs
            enabled: Whether TensorBoard logging is enabled
        """
        self.log_dir = log_dir
        self.enabled = enabled
        self.writer = None
        self.step = 0
        
        if self.enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=log_dir)
            except ImportError:
                print("⚠️  TensorBoard not available. Install with: pip install tensorboard")
                self.enabled = False
    
    def execute(self, context, next_callable):
        result = next_callable(context)
        
        if not self.enabled or self.writer is None:
            return result
        
        event_name = context.get('_current_event', '')
        
        # Log loss after loss calculation
        if 'CalculateLoss' in event_name:
            loss_value = context.get('loss_value')
            if loss_value is not None:
                self.writer.add_scalar('Loss/train', loss_value, self.step)
        
        # Log learning rate after weight update
        if 'UpdateWeights' in event_name:
            lr = context.get('learning_rate')
            if lr is not None:
                self.writer.add_scalar('Learning_Rate', lr, self.step)
            self.step += 1
        
        # Log validation metrics
        if 'Validation' in event_name:
            val_loss = context.get('val_loss')
            val_accuracy = context.get('val_accuracy')
            
            if val_loss is not None:
                self.writer.add_scalar('Loss/validation', val_loss, self.step)
            if val_accuracy is not None:
                self.writer.add_scalar('Accuracy/validation', val_accuracy, self.step)
        
        # Log gradient norms
        gradient_norms = context.get('gradient_norms', {})
        if gradient_norms:
            for name, norm in list(gradient_norms.items())[:5]:  # Log first 5 layers
                clean_name = name.replace('.', '/')
                self.writer.add_scalar(f'Gradients/{clean_name}', norm, self.step)
        
        return result
    
    def close(self):
        """Close the TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()


class EarlyStoppingMiddleware(Middleware):
    """
    Implement early stopping based on validation loss.
    
    Stops training if validation loss doesn't improve for a specified
    number of epochs (patience).
    """
    
    def __init__(self, patience=5, min_delta=0.001, verbose=True):
        """
        Initialize the EarlyStoppingMiddleware.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False
    
    def execute(self, context, next_callable):
        result = next_callable(context)
        
        # Only check after validation
        event_name = context.get('_current_event', '')
        if 'Validation' not in event_name:
            return result
        
        val_loss = context.get('val_loss')
        if val_loss is None:
            return result
        
        # Check if loss improved
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"✓ Validation loss improved to {val_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"⚠️  No improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                self.should_stop = True
                context.set('early_stop', True)
                if self.verbose:
                    print(f"🛑 Early stopping triggered! Best loss: {self.best_loss:.4f}")
        
        return result
    
    def should_stop_training(self):
        """Check if training should stop."""
        return self.should_stop
    
    def reset(self):
        """Reset the early stopping state."""
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

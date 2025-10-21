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


class AuditLogMiddleware(Middleware):
    """
    Create comprehensive audit logs for ML training experiments.
    
    Logs every training decision, hyperparameter, metric, and event for:
    - Experiment reproducibility
    - Compliance and governance
    - Debugging and analysis
    - Model lineage tracking
    """
    
    def __init__(self, log_file='audit_log.jsonl', log_to_console=False):
        """
        Initialize the AuditLogMiddleware.
        
        Args:
            log_file: Path to JSONL audit log file
            log_to_console: Whether to also print logs to console
        """
        self.log_file = log_file
        self.log_to_console = log_to_console
        self.session_id = None
        self.event_counter = 0
        
        import uuid
        import datetime
        self.session_id = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Initialize log file
        self._write_log({
            'type': 'session_start',
            'session_id': self.session_id,
            'timestamp': datetime.datetime.now().isoformat(),
        })
    
    def execute(self, context, next_callable):
        import datetime
        import json
        
        event_name = context.get('_current_event', 'Unknown')
        self.event_counter += 1
        
        pre_log = {
            'type': 'event_start',
            'session_id': self.session_id,
            'event_id': self.event_counter,
            'event_name': event_name,
            'timestamp': datetime.datetime.now().isoformat(),
        }
        
        context_snapshot = {}
        for key in context.keys():
            if not key.startswith('_') and key not in ['batch', 'output', 'model', 'optimizer', 'criterion']:
                value = context.get(key)
                if isinstance(value, (int, float, str, bool, list, dict)):
                    context_snapshot[key] = value
                elif hasattr(value, 'item'):  # PyTorch scalar tensor
                    try:
                        context_snapshot[key] = value.item()
                    except:
                        pass
        
        pre_log['context'] = context_snapshot
        self._write_log(pre_log)
        
        start_time = datetime.datetime.now()
        result = next_callable(context)
        end_time = datetime.datetime.now()
        
        post_log = {
            'type': 'event_complete',
            'session_id': self.session_id,
            'event_id': self.event_counter,
            'event_name': event_name,
            'timestamp': end_time.isoformat(),
            'duration_ms': (end_time - start_time).total_seconds() * 1000,
            'success': result.success,
        }
        
        if not result.success:
            post_log['error'] = result.error
        
        post_context = {}
        for key in context.keys():
            if not key.startswith('_') and key not in ['batch', 'output', 'model', 'optimizer', 'criterion']:
                value = context.get(key)
                if isinstance(value, (int, float, str, bool, list, dict)):
                    post_context[key] = value
                elif hasattr(value, 'item'):
                    try:
                        post_context[key] = value.item()
                    except:
                        pass
        
        post_log['context'] = post_context
        self._write_log(post_log)
        
        return result
    
    def _write_log(self, log_entry):
        """Write a log entry to the audit log file."""
        import json
        
        if self.log_to_console:
            print(f"[AUDIT] {json.dumps(log_entry)}")
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"⚠️  Failed to write audit log: {e}")
    
    def close(self):
        """Close the audit log session."""
        import datetime
        self._write_log({
            'type': 'session_end',
            'session_id': self.session_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'total_events': self.event_counter,
        })


class ValidationMiddleware(Middleware):
    """
    Validate tensor shapes, data ranges, and model states during training.
    
    Catches common issues:
    - Shape mismatches
    - NaN or Inf values in tensors
    - Out-of-range values
    - Invalid model states
    """
    
    def __init__(self, strict=True, verbose=True):
        """
        Initialize the ValidationMiddleware.
        
        Args:
            strict: If True, fail on validation errors. If False, warn only.
            verbose: Whether to print validation warnings
        """
        self.strict = strict
        self.verbose = verbose
        self.validation_errors = []
    
    def execute(self, context, next_callable):
        import torch
        from eventchains import Result
        
        event_name = context.get('_current_event', 'Unknown')
        
        # Pre-execution validation
        pre_errors = self._validate_context(context, event_name, 'pre')
        
        if pre_errors and self.strict:
            error_msg = f"Pre-validation failed for {event_name}: {'; '.join(pre_errors)}"
            if self.verbose:
                print(f"❌ {error_msg}")
            return Result.fail(error_msg)
        
        result = next_callable(context)
        
        # Post-execution validation
        post_errors = self._validate_context(context, event_name, 'post')
        
        if post_errors:
            self.validation_errors.extend(post_errors)
            if self.strict:
                error_msg = f"Post-validation failed for {event_name}: {'; '.join(post_errors)}"
                if self.verbose:
                    print(f"❌ {error_msg}")
                return Result.fail(error_msg)
        
        return result
    
    def _validate_context(self, context, event_name, phase):
        """Validate context data."""
        import torch
        errors = []
        
        # Validate tensors in context
        for key in ['batch', 'labels', 'output', 'loss']:
            if context.has(key):
                tensor = context.get(key)
                if isinstance(tensor, torch.Tensor):
                    # Check for NaN
                    if torch.isnan(tensor).any():
                        error = f"{phase}: NaN detected in '{key}'"
                        errors.append(error)
                        if self.verbose:
                            print(f"⚠️  {error}")
                    
                    # Check for Inf
                    if torch.isinf(tensor).any():
                        error = f"{phase}: Inf detected in '{key}'"
                        errors.append(error)
                        if self.verbose:
                            print(f"⚠️  {error}")
                    
                    if key == 'batch' and context.has('labels'):
                        labels = context.get('labels')
                        if isinstance(labels, torch.Tensor):
                            if tensor.size(0) != labels.size(0):
                                error = f"{phase}: Batch size mismatch - batch: {tensor.size(0)}, labels: {labels.size(0)}"
                                errors.append(error)
                                if self.verbose:
                                    print(f"⚠️  {error}")
        
        # Validate gradients after backpropagation
        if 'Backpropagation' in event_name and phase == 'post':
            gradient_norms = context.get('gradient_norms', {})
            for param_name, norm in gradient_norms.items():
                if norm != norm or norm == float('inf'):  # NaN or Inf
                    error = f"{phase}: Invalid gradient in {param_name}"
                    errors.append(error)
                    if self.verbose:
                        print(f"⚠️  {error}")
        
        return errors
    
    def get_errors(self):
        """Return all validation errors."""
        return self.validation_errors.copy()
    
    def clear_errors(self):
        """Clear validation errors."""
        self.validation_errors.clear()


class MetricsCollectorMiddleware(Middleware):
    """
    Collect and export metrics for monitoring systems (Prometheus, StatsD, etc.).
    
    Tracks:
    - Event execution counts
    - Event durations
    - Success/failure rates
    - Custom ML metrics (loss, accuracy, etc.)
    """
    
    def __init__(self, export_format='prometheus', export_file='metrics.txt'):
        """
        Initialize the MetricsCollectorMiddleware.
        
        Args:
            export_format: Format for metrics export ('prometheus', 'statsd', 'json')
            export_file: File to export metrics to
        """
        self.export_format = export_format
        self.export_file = export_file
        
        self.event_counts = {}
        self.event_durations = {}
        self.event_failures = {}
        self.custom_metrics = {}
        
        import time
        self.start_time = time.time()
    
    def execute(self, context, next_callable):
        import time
        
        event_name = context.get('_current_event', 'Unknown')
        
        # Initialize counters
        if event_name not in self.event_counts:
            self.event_counts[event_name] = 0
            self.event_durations[event_name] = []
            self.event_failures[event_name] = 0
        
        # Time execution
        start = time.perf_counter()
        result = next_callable(context)
        duration = time.perf_counter() - start
        
        self.event_counts[event_name] += 1
        self.event_durations[event_name].append(duration)
        
        if not result.success:
            self.event_failures[event_name] += 1
        
        if context.has('loss_value'):
            self.custom_metrics['loss'] = context.get('loss_value')
        
        if context.has('val_accuracy'):
            self.custom_metrics['val_accuracy'] = context.get('val_accuracy')
        
        if context.has('learning_rate'):
            self.custom_metrics['learning_rate'] = context.get('learning_rate')
        
        if context.has('total_gradient_norm'):
            self.custom_metrics['gradient_norm'] = context.get('total_gradient_norm')
        
        return result
    
    def export_metrics(self):
        """Export metrics to file in specified format."""
        if self.export_format == 'prometheus':
            self._export_prometheus()
        elif self.export_format == 'json':
            self._export_json()
        elif self.export_format == 'statsd':
            self._export_statsd()
    
    def _export_prometheus(self):
        """Export metrics in Prometheus format."""
        import time
        
        lines = []
        lines.append("# HELP eventchains_event_count Total number of event executions")
        lines.append("# TYPE eventchains_event_count counter")
        
        for event, count in self.event_counts.items():
            lines.append(f'eventchains_event_count{{event="{event}"}} {count}')
        
        lines.append("\n# HELP eventchains_event_duration_seconds Event execution duration")
        lines.append("# TYPE eventchains_event_duration_seconds histogram")
        
        for event, durations in self.event_durations.items():
            if durations:
                avg_duration = sum(durations) / len(durations)
                lines.append(f'eventchains_event_duration_seconds{{event="{event}"}} {avg_duration}')
        
        lines.append("\n# HELP eventchains_event_failures Total number of event failures")
        lines.append("# TYPE eventchains_event_failures counter")
        
        for event, failures in self.event_failures.items():
            lines.append(f'eventchains_event_failures{{event="{event}"}} {failures}')
        
        if self.custom_metrics:
            lines.append("\n# HELP eventchains_ml_metric ML-specific metrics")
            lines.append("# TYPE eventchains_ml_metric gauge")
            
            for metric_name, value in self.custom_metrics.items():
                lines.append(f'eventchains_ml_metric{{metric="{metric_name}"}} {value}')
        
        try:
            with open(self.export_file, 'w') as f:
                f.write('\n'.join(lines))
        except Exception as e:
            print(f"⚠️  Failed to export metrics: {e}")
    
    def _export_json(self):
        """Export metrics in JSON format."""
        import json
        import time
        
        metrics = {
            'timestamp': time.time(),
            'uptime_seconds': time.time() - self.start_time,
            'events': {},
            'custom_metrics': self.custom_metrics,
        }
        
        for event in self.event_counts.keys():
            durations = self.event_durations[event]
            metrics['events'][event] = {
                'count': self.event_counts[event],
                'failures': self.event_failures[event],
                'avg_duration_seconds': sum(durations) / len(durations) if durations else 0,
                'min_duration_seconds': min(durations) if durations else 0,
                'max_duration_seconds': max(durations) if durations else 0,
            }
        
        try:
            with open(self.export_file, 'w') as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            print(f"⚠️  Failed to export metrics: {e}")
    
    def _export_statsd(self):
        """Export metrics in StatsD format."""
        lines = []
        
        for event, count in self.event_counts.items():
            lines.append(f"eventchains.event.count.{event}:{count}|c")
        
        for event, durations in self.event_durations.items():
            if durations:
                avg_duration = sum(durations) / len(durations) * 1000  # Convert to ms
                lines.append(f"eventchains.event.duration.{event}:{avg_duration:.2f}|ms")
        
        for event, failures in self.event_failures.items():
            if failures > 0:
                lines.append(f"eventchains.event.failures.{event}:{failures}|c")
        
        try:
            with open(self.export_file, 'w') as f:
                f.write('\n'.join(lines))
        except Exception as e:
            print(f"⚠️  Failed to export metrics: {e}")
    
    def get_summary(self):
        """Get a summary of collected metrics."""
        summary = {
            'total_events': sum(self.event_counts.values()),
            'total_failures': sum(self.event_failures.values()),
            'events': {},
        }
        
        for event in self.event_counts.keys():
            durations = self.event_durations[event]
            summary['events'][event] = {
                'count': self.event_counts[event],
                'failures': self.event_failures[event],
                'avg_duration_ms': sum(durations) / len(durations) * 1000 if durations else 0,
            }
        
        return summary


class CompressionMiddleware(Middleware):
    """
    Compress large tensor data in context to reduce memory usage.
    
    Useful for:
    - Storing activation maps for later analysis
    - Keeping gradient history
    - Logging large intermediate results
    - Memory-constrained environments
    """
    
    def __init__(self, compress_keys=None, compression_level=6, threshold_mb=1.0):
        """
        Initialize the CompressionMiddleware.
        
        Args:
            compress_keys: List of context keys to compress (None = auto-detect large tensors)
            compression_level: Compression level 0-9 (higher = more compression, slower)
            threshold_mb: Only compress tensors larger than this size in MB
        """
        self.compress_keys = compress_keys or []
        self.compression_level = compression_level
        self.threshold_bytes = threshold_mb * 1024 * 1024
        self.compression_stats = {
            'compressed_count': 0,
            'original_bytes': 0,
            'compressed_bytes': 0,
        }
    
    def execute(self, context, next_callable):
        import torch
        import pickle
        import gzip
        
        result = next_callable(context)
        
        keys_to_compress = self.compress_keys if self.compress_keys else []
        
        if not keys_to_compress:
            for key in list(context.keys()):
                if not key.startswith('_'):
                    value = context.get(key)
                    if isinstance(value, torch.Tensor):
                        tensor_bytes = value.element_size() * value.nelement()
                        if tensor_bytes > self.threshold_bytes:
                            keys_to_compress.append(key)
        
        for key in keys_to_compress:
            if context.has(key):
                value = context.get(key)
                if isinstance(value, torch.Tensor):
                    try:
                        tensor_cpu = value.detach().cpu()
                        original_bytes = tensor_cpu.element_size() * tensor_cpu.nelement()
                        
                        serialized = pickle.dumps(tensor_cpu.numpy())
                        compressed = gzip.compress(serialized, compresslevel=self.compression_level)
                        
                        compressed_data = {
                            '_compressed': True,
                            'data': compressed,
                            'shape': tensor_cpu.shape,
                            'dtype': str(tensor_cpu.dtype),
                            'original_bytes': original_bytes,
                            'compressed_bytes': len(compressed),
                        }
                        
                        context.set(f'{key}_compressed', compressed_data)
                        
                        self.compression_stats['compressed_count'] += 1
                        self.compression_stats['original_bytes'] += original_bytes
                        self.compression_stats['compressed_bytes'] += len(compressed)
                        
                    except Exception as e:
                        print(f"⚠️  Failed to compress {key}: {e}")
        
        return result
    
    def decompress(self, compressed_data):
        """
        Decompress data that was compressed by this middleware.
        
        Args:
            compressed_data: Dictionary with compressed data and metadata
            
        Returns:
            Original tensor
        """
        import torch
        import pickle
        import gzip
        import numpy as np
        
        if not isinstance(compressed_data, dict) or not compressed_data.get('_compressed'):
            raise ValueError("Data is not compressed by CompressionMiddleware")
        
        try:
            decompressed = gzip.decompress(compressed_data['data'])
            numpy_array = pickle.loads(decompressed)
            
            tensor = torch.from_numpy(numpy_array)
            
            return tensor
        except Exception as e:
            raise RuntimeError(f"Failed to decompress data: {e}")
    
    def get_stats(self):
        """Get compression statistics."""
        stats = self.compression_stats.copy()
        
        if stats['original_bytes'] > 0:
            stats['compression_ratio'] = stats['compressed_bytes'] / stats['original_bytes']
            stats['space_saved_mb'] = (stats['original_bytes'] - stats['compressed_bytes']) / (1024 * 1024)
            stats['space_saved_percent'] = (1 - stats['compression_ratio']) * 100
        else:
            stats['compression_ratio'] = 1.0
            stats['space_saved_mb'] = 0.0
            stats['space_saved_percent'] = 0.0
        
        return stats
    
    def print_stats(self):
        """Print compression statistics."""
        stats = self.get_stats()
        
        print("\n" + "=" * 80)
        print("Compression Statistics")
        print("=" * 80)
        print(f"Tensors compressed: {stats['compressed_count']}")
        print(f"Original size: {stats['original_bytes'] / (1024 * 1024):.2f} MB")
        print(f"Compressed size: {stats['compressed_bytes'] / (1024 * 1024):.2f} MB")
        print(f"Space saved: {stats['space_saved_mb']:.2f} MB ({stats['space_saved_percent']:.1f}%)")
        print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
        print("=" * 80)

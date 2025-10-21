"""
Tests for production middleware: AuditLog, Validation, MetricsCollector, Compression
"""

import unittest
import tempfile
import os
import json
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from eventchains import EventChain, ChainableEvent, EventContext, Result
from eventchains_ml import (
    AuditLogMiddleware,
    ValidationMiddleware,
    MetricsCollectorMiddleware,
    CompressionMiddleware,
)


class DummyEvent(ChainableEvent):
    """Simple event for testing."""
    
    def __init__(self, should_fail=False, set_tensor=False, large_tensor=False):
        self.should_fail = should_fail
        self.set_tensor = set_tensor
        self.large_tensor = large_tensor
    
    def execute(self, context):
        if self.set_tensor:
            if self.large_tensor:
                tensor = torch.randn(1000, 1000)  # ~4MB tensor
            else:
                tensor = torch.randn(10, 10)
            context.set('test_tensor', tensor)
        
        if self.should_fail:
            return Result.fail("Test failure")
        
        context.set('test_value', 42)
        return Result.ok()


class TestAuditLogMiddleware(unittest.TestCase):
    """Test AuditLogMiddleware functionality."""
    
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.jsonl')
        self.temp_file.close()
        self.log_file = self.temp_file.name
    
    def tearDown(self):
        if os.path.exists(self.log_file):
            os.unlink(self.log_file)
    
    def test_audit_log_creation(self):
        """Test audit log middleware initialization."""
        audit_log = AuditLogMiddleware(log_file=self.log_file)
        
        self.assertIsNotNone(audit_log.session_id)
        self.assertEqual(audit_log.event_counter, 0)
        self.assertEqual(audit_log.log_file, self.log_file)
        
        with open(self.log_file, 'r') as f:
            line = f.readline().strip()
            log_entry = json.loads(line)
            self.assertEqual(log_entry['type'], 'session_start')
            self.assertEqual(log_entry['session_id'], audit_log.session_id)
    
    def test_audit_log_event_execution(self):
        """Test event execution logging."""
        audit_log = AuditLogMiddleware(log_file=self.log_file)
        
        chain = (EventChain()
            .add_event(DummyEvent())
            .use_middleware(audit_log))
        
        context = EventContext({'input': 'test'})
        result = chain.execute(context)
        
        self.assertTrue(result.success)
        
        with open(self.log_file, 'r') as f:
            lines = f.readlines()
        
        self.assertEqual(len(lines), 3)
        
        event_start = json.loads(lines[1])
        self.assertEqual(event_start['type'], 'event_start')
        self.assertEqual(event_start['event_name'], 'DummyEvent')
        self.assertEqual(event_start['context']['input'], 'test')
        
        event_complete = json.loads(lines[2])
        self.assertEqual(event_complete['type'], 'event_complete')
        self.assertEqual(event_complete['event_name'], 'DummyEvent')
        self.assertTrue(event_complete['success'])
        self.assertEqual(event_complete['context']['test_value'], 42)
    
    def test_audit_log_failure(self):
        """Test logging of failed events."""
        audit_log = AuditLogMiddleware(log_file=self.log_file)
        
        chain = (EventChain()
            .add_event(DummyEvent(should_fail=True))
            .use_middleware(audit_log))
        
        context = EventContext()
        result = chain.execute(context)
        
        self.assertFalse(result.success)
        
        with open(self.log_file, 'r') as f:
            lines = f.readlines()
        
        event_complete = json.loads(lines[2])
        self.assertFalse(event_complete['success'])
        self.assertEqual(event_complete['error'], 'Test failure')
    
    def test_audit_log_close(self):
        """Test audit log session closing."""
        audit_log = AuditLogMiddleware(log_file=self.log_file)
        audit_log.close()
        
        with open(self.log_file, 'r') as f:
            lines = f.readlines()
        
        self.assertEqual(len(lines), 2)
        
        session_end = json.loads(lines[1])
        self.assertEqual(session_end['type'], 'session_end')
        self.assertEqual(session_end['total_events'], 0)


class TestValidationMiddleware(unittest.TestCase):
    """Test ValidationMiddleware functionality."""
    
    def test_validation_middleware_creation(self):
        """Test validation middleware initialization."""
        validation = ValidationMiddleware(strict=True, verbose=False)
        
        self.assertTrue(validation.strict)
        self.assertFalse(validation.verbose)
        self.assertEqual(len(validation.validation_errors), 0)
    
    def test_validation_success(self):
        """Test validation with valid data."""
        validation = ValidationMiddleware(strict=True, verbose=False)
        
        chain = (EventChain()
            .add_event(DummyEvent())
            .use_middleware(validation))
        
        context = EventContext({
            'batch': torch.randn(32, 1, 28, 28),
            'labels': torch.randint(0, 10, (32,))
        })
        
        result = chain.execute(context)
        
        self.assertTrue(result.success)
        self.assertEqual(len(validation.get_errors()), 0)
    
    def test_validation_nan_detection(self):
        """Test NaN detection in tensors."""
        validation = ValidationMiddleware(strict=True, verbose=False)
        
        chain = (EventChain()
            .add_event(DummyEvent())
            .use_middleware(validation))
        
        batch = torch.randn(32, 1, 28, 28)
        batch[0, 0, 0, 0] = float('nan')
        
        context = EventContext({
            'batch': batch,
            'labels': torch.randint(0, 10, (32,))
        })
        
        result = chain.execute(context)
        
        self.assertFalse(result.success)
        self.assertIn("NaN detected", result.error)
    
    def test_validation_inf_detection(self):
        """Test Inf detection in tensors."""
        validation = ValidationMiddleware(strict=True, verbose=False)
        
        chain = (EventChain()
            .add_event(DummyEvent())
            .use_middleware(validation))
        
        batch = torch.randn(32, 1, 28, 28)
        batch[0, 0, 0, 0] = float('inf')
        
        context = EventContext({
            'batch': batch,
            'labels': torch.randint(0, 10, (32,))
        })
        
        result = chain.execute(context)
        
        self.assertFalse(result.success)
        self.assertIn("Inf detected", result.error)
    
    def test_validation_shape_mismatch(self):
        """Test batch/labels shape mismatch detection."""
        validation = ValidationMiddleware(strict=True, verbose=False)
        
        chain = (EventChain()
            .add_event(DummyEvent())
            .use_middleware(validation))
        
        context = EventContext({
            'batch': torch.randn(32, 1, 28, 28),
            'labels': torch.randint(0, 10, (16,))  # Wrong batch size
        })
        
        result = chain.execute(context)
        
        self.assertFalse(result.success)
        self.assertIn("Batch size mismatch", result.error)
    
    def test_validation_lenient_mode(self):
        """Test validation in lenient mode."""
        validation = ValidationMiddleware(strict=False, verbose=False)
        
        chain = (EventChain()
            .add_event(DummyEvent())
            .use_middleware(validation))
        
        batch = torch.randn(32, 1, 28, 28)
        batch[0, 0, 0, 0] = float('nan')
        
        context = EventContext({
            'batch': batch,
            'labels': torch.randint(0, 10, (32,))
        })
        
        result = chain.execute(context)
        
        self.assertTrue(result.success)
        errors = validation.get_errors()
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("NaN detected" in error for error in errors))


class TestMetricsCollectorMiddleware(unittest.TestCase):
    """Test MetricsCollectorMiddleware functionality."""
    
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt')
        self.temp_file.close()
        self.metrics_file = self.temp_file.name
    
    def tearDown(self):
        if os.path.exists(self.metrics_file):
            os.unlink(self.metrics_file)
    
    def test_metrics_collector_creation(self):
        """Test metrics collector initialization."""
        metrics = MetricsCollectorMiddleware(
            export_format='prometheus',
            export_file=self.metrics_file
        )
        
        self.assertEqual(metrics.export_format, 'prometheus')
        self.assertEqual(metrics.export_file, self.metrics_file)
        self.assertEqual(len(metrics.event_counts), 0)
    
    def test_metrics_collection(self):
        """Test basic metrics collection."""
        metrics = MetricsCollectorMiddleware(export_file=self.metrics_file)
        
        chain = (EventChain()
            .add_event(DummyEvent())
            .use_middleware(metrics))
        
        context = EventContext()
        result = chain.execute(context)
        
        self.assertTrue(result.success)
        
        self.assertEqual(metrics.event_counts['DummyEvent'], 1)
        self.assertEqual(len(metrics.event_durations['DummyEvent']), 1)
        self.assertEqual(metrics.event_failures['DummyEvent'], 0)
    
    def test_metrics_failure_tracking(self):
        """Test failure tracking in metrics."""
        metrics = MetricsCollectorMiddleware(export_file=self.metrics_file)
        
        chain = (EventChain()
            .add_event(DummyEvent(should_fail=True))
            .use_middleware(metrics))
        
        context = EventContext()
        result = chain.execute(context)
        
        self.assertFalse(result.success)
        
        self.assertEqual(metrics.event_failures['DummyEvent'], 1)
    
    def test_metrics_ml_values(self):
        """Test ML-specific metrics collection."""
        metrics = MetricsCollectorMiddleware(export_file=self.metrics_file)
        
        chain = (EventChain()
            .add_event(DummyEvent())
            .use_middleware(metrics))
        
        context = EventContext({
            'loss_value': 0.5,
            'val_accuracy': 95.0,
            'learning_rate': 0.001,
            'total_gradient_norm': 2.5
        })
        
        result = chain.execute(context)
        
        self.assertTrue(result.success)
        
        self.assertEqual(metrics.custom_metrics['loss'], 0.5)
        self.assertEqual(metrics.custom_metrics['val_accuracy'], 95.0)
        self.assertEqual(metrics.custom_metrics['learning_rate'], 0.001)
        self.assertEqual(metrics.custom_metrics['gradient_norm'], 2.5)
    
    def test_prometheus_export(self):
        """Test Prometheus format export."""
        metrics = MetricsCollectorMiddleware(
            export_format='prometheus',
            export_file=self.metrics_file
        )
        
        chain = (EventChain()
            .add_event(DummyEvent())
            .use_middleware(metrics))
        
        context = EventContext({'loss_value': 0.5})
        chain.execute(context)
        
        metrics.export_metrics()
        
        with open(self.metrics_file, 'r') as f:
            content = f.read()
        
        self.assertIn('# HELP eventchains_event_count', content)
        self.assertIn('# TYPE eventchains_event_count counter', content)
        self.assertIn('eventchains_event_count{event="DummyEvent"} 1', content)
        self.assertIn('eventchains_ml_metric{metric="loss"} 0.5', content)
    
    def test_json_export(self):
        """Test JSON format export."""
        metrics = MetricsCollectorMiddleware(
            export_format='json',
            export_file=self.metrics_file
        )
        
        chain = (EventChain()
            .add_event(DummyEvent())
            .use_middleware(metrics))
        
        context = EventContext()
        chain.execute(context)
        
        metrics.export_metrics()
        
        with open(self.metrics_file, 'r') as f:
            data = json.load(f)
        
        self.assertIn('timestamp', data)
        self.assertIn('events', data)
        self.assertIn('DummyEvent', data['events'])
        self.assertEqual(data['events']['DummyEvent']['count'], 1)


class TestCompressionMiddleware(unittest.TestCase):
    """Test CompressionMiddleware functionality."""
    
    def test_compression_middleware_creation(self):
        """Test compression middleware initialization."""
        compression = CompressionMiddleware(
            compress_keys=['test_tensor'],
            compression_level=6,
            threshold_mb=1.0
        )
        
        self.assertEqual(compression.compress_keys, ['test_tensor'])
        self.assertEqual(compression.compression_level, 6)
        self.assertEqual(compression.threshold_bytes, 1024 * 1024)
    
    def test_compression_specific_keys(self):
        """Test compression of specific keys."""
        compression = CompressionMiddleware(
            compress_keys=['test_tensor'],
            threshold_mb=0.001  # Very low threshold
        )
        
        chain = (EventChain()
            .add_event(DummyEvent(set_tensor=True))
            .use_middleware(compression))
        
        context = EventContext()
        result = chain.execute(context)
        
        self.assertTrue(result.success)
        
        self.assertTrue(context.has('test_tensor_compressed'))
        compressed_data = context.get('test_tensor_compressed')
        self.assertTrue(compressed_data['_compressed'])
        self.assertEqual(compressed_data['shape'], (10, 10))
        
        stats = compression.get_stats()
        self.assertEqual(stats['compressed_count'], 1)
        self.assertGreater(stats['original_bytes'], 0)
        self.assertGreater(stats['compressed_bytes'], 0)
    
    def test_compression_auto_detect(self):
        """Test auto-detection of large tensors."""
        compression = CompressionMiddleware(
            compress_keys=[],  # Auto-detect
            threshold_mb=0.5  # 0.5 MB threshold
        )
        
        chain = (EventChain()
            .add_event(DummyEvent(set_tensor=True, large_tensor=True))
            .use_middleware(compression))
        
        context = EventContext()
        result = chain.execute(context)
        
        self.assertTrue(result.success)
        
        self.assertTrue(context.has('test_tensor_compressed'))
        
        stats = compression.get_stats()
        self.assertEqual(stats['compressed_count'], 1)
    
    def test_compression_threshold(self):
        """Test compression threshold."""
        compression = CompressionMiddleware(
            compress_keys=[],
            threshold_mb=10.0  # High threshold - won't compress small tensor
        )
        
        chain = (EventChain()
            .add_event(DummyEvent(set_tensor=True))
            .use_middleware(compression))
        
        context = EventContext()
        result = chain.execute(context)
        
        self.assertTrue(result.success)
        
        self.assertFalse(context.has('test_tensor_compressed'))
        
        stats = compression.get_stats()
        self.assertEqual(stats['compressed_count'], 0)
    
    def test_decompression(self):
        """Test decompression of compressed data."""
        compression = CompressionMiddleware(
            compress_keys=['test_tensor'],
            threshold_mb=0.001
        )
        
        chain = (EventChain()
            .add_event(DummyEvent(set_tensor=True, large_tensor=True))
            .use_middleware(compression))
        
        context = EventContext()
        result = chain.execute(context)
        
        self.assertTrue(result.success)
        
        original_tensor = context.get('test_tensor')
        compressed_data = context.get('test_tensor_compressed')
        
        decompressed_tensor = compression.decompress(compressed_data)
        
        self.assertTrue(torch.allclose(original_tensor, decompressed_tensor))
        self.assertEqual(original_tensor.shape, decompressed_tensor.shape)
    
    def test_compression_stats(self):
        """Test compression statistics calculation."""
        compression = CompressionMiddleware(
            compress_keys=['test_tensor'],
            threshold_mb=0.001
        )
        
        chain = (EventChain()
            .add_event(DummyEvent(set_tensor=True, large_tensor=True))
            .use_middleware(compression))
        
        context = EventContext()
        chain.execute(context)
        
        stats = compression.get_stats()
        
        self.assertEqual(stats['compressed_count'], 1)
        self.assertGreater(stats['original_bytes'], 0)
        self.assertGreater(stats['compressed_bytes'], 0)
        self.assertLess(stats['compression_ratio'], 1.0)
        self.assertGreater(stats['space_saved_percent'], 0)
        self.assertGreater(stats['space_saved_mb'], 0)


if __name__ == '__main__':
    unittest.main()

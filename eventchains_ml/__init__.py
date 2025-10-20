"""
EventChains ML - Machine Learning Extension for EventChains

Provides events and middleware specifically designed for machine learning workflows,
with a focus on bringing observability and debuggability to neural network training.

Features:
- PyTorch integration events (forward pass, backprop, weight updates)
- Diagnostic middleware (gradient monitoring, dead neuron detection)
- Performance profiling for ML workflows
- TensorBoard integration
- Minimal overhead design for high-iteration training
"""

__version__ = "1.0.0"
__author__ = "EventChains Contributors"

from .events import (
    LoadBatchEvent,
    ForwardPassEvent,
    CalculateLossEvent,
    BackpropagationEvent,
    UpdateWeightsEvent,
    ValidationEvent,
)

from .middleware import (
    GradientMonitorMiddleware,
    DeadNeuronDetectorMiddleware,
    PerformanceProfilerMiddleware,
    TensorBoardMiddleware,
    EarlyStoppingMiddleware,
)

__all__ = [
    # Events
    'LoadBatchEvent',
    'ForwardPassEvent',
    'CalculateLossEvent',
    'BackpropagationEvent',
    'UpdateWeightsEvent',
    'ValidationEvent',
    
    # Middleware
    'GradientMonitorMiddleware',
    'DeadNeuronDetectorMiddleware',
    'PerformanceProfilerMiddleware',
    'TensorBoardMiddleware',
    'EarlyStoppingMiddleware',
]

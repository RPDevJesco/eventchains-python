# EventChains - Python Implementation

A universal design pattern for building sequential workflows with composable middleware.

## Overview

EventChains is a design pattern that orchestrates sequential execution of discrete units of work (events) through a configurable pipeline of cross-cutting concerns (middleware). It provides a structured approach to building workflows that are:

- **Observable** - Every step is visible and loggable
- **Composable** - Mix and match events and middleware
- **Testable** - Events can be tested in isolation
- **Maintainable** - Clear separation of concerns
- **Performant** - Minimal overhead, pipeline built once

## Core Concepts

### EventContext
A shared data container (dictionary) that flows through the entire chain, enabling communication between sequential events.

### ChainableEvent
A discrete unit of business logic that:
- Receives context and can read/modify it
- Returns a Result indicating success or failure
- Should be stateless (all state in context)
- Is testable in isolation

### Middleware
Wraps event execution with cross-cutting concerns:
- Executes in LIFO order (reverse of registration)
- Can run logic before, after, or around events
- Examples: logging, timing, error handling, validation

### EventChain
Orchestrates the sequential flow:
- Manages middleware pipeline construction
- Handles error propagation based on fault tolerance
- Provides lifecycle management

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from eventchains import EventChain, ChainableEvent, EventContext, Result, Middleware

# Define events
class ValidateInput(ChainableEvent):
    def execute(self, context):
        value = context.get('input')
        if value is None or value < 0:
            return Result.fail("Invalid input")
        return Result.ok()

class ProcessData(ChainableEvent):
    def execute(self, context):
        value = context.get('input')
        result = value * 2
        context.set('output', result)
        return Result.ok()

# Define middleware
class LoggingMiddleware(Middleware):
    def execute(self, context, next_callable):
        event_name = context.get('_current_event', 'Unknown')
        print(f"Starting {event_name}")
        
        result = next_callable(context)
        
        if result.success:
            print(f"Completed {event_name}")
        else:
            print(f"Failed {event_name}: {result.error}")
        
        return result

# Build and execute chain
chain = (EventChain()
    .add_event(ValidateInput())
    .add_event(ProcessData())
    .use_middleware(LoggingMiddleware()))

context = EventContext({'input': 5})
result = chain.execute(context)

if result.success:
    print(f"Output: {context.get('output')}")  # Output: 10
```

## Examples

### Basic Order Processing

```python
class ValidateOrder(ChainableEvent):
    def execute(self, context):
        order = context.get('order')
        if not order or not order.get('items'):
            return Result.fail("Invalid order")
        return Result.ok()

class CalculateTotals(ChainableEvent):
    def execute(self, context):
        order = context.get('order')
        total = sum(item['price'] * item['quantity'] 
                   for item in order['items'])
        context.set('total', total)
        return Result.ok()

class ProcessPayment(ChainableEvent):
    def execute(self, context):
        total = context.get('total')
        # Payment processing logic here
        context.set('payment_id', 'PAY-12345')
        return Result.ok()

# Build chain
order_chain = (EventChain()
    .add_event(ValidateOrder())
    .add_event(CalculateTotals())
    .add_event(ProcessPayment())
    .use_middleware(LoggingMiddleware()))

# Execute
context = EventContext({
    'order': {
        'items': [
            {'name': 'Widget', 'price': 10.00, 'quantity': 2},
            {'name': 'Gadget', 'price': 15.00, 'quantity': 1}
        ]
    }
})

result = order_chain.execute(context)
```

### Performance Profiling Middleware

```python
import time

class ProfilingMiddleware(Middleware):
    def __init__(self):
        self.timings = {}
    
    def execute(self, context, next_callable):
        event_name = context.get('_current_event', 'Unknown')
        
        start = time.perf_counter()
        result = next_callable(context)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        
        if event_name not in self.timings:
            self.timings[event_name] = []
        self.timings[event_name].append(elapsed)
        
        context.set(f'{event_name}_time_ms', elapsed)
        return result
    
    def report(self):
        for event, times in self.timings.items():
            avg = sum(times) / len(times)
            print(f"{event}: avg={avg:.2f}ms, calls={len(times)}")
```

## Fault Tolerance

EventChains supports different fault tolerance modes:

```python
from eventchains import FaultTolerance

# STRICT - Any failure stops the chain (default)
chain = EventChain(fault_tolerance=FaultTolerance.STRICT)

# LENIENT - Non-critical failures continue
chain = EventChain(fault_tolerance=FaultTolerance.LENIENT)

# BEST_EFFORT - All events attempted regardless of failures
chain = EventChain(fault_tolerance=FaultTolerance.BEST_EFFORT)
```

## Design Principles

1. **Minimal Overhead** - No decorators, pipeline built once and cached
2. **Explicit over Implicit** - Middleware registered explicitly on chain
3. **Stateless Events** - All state flows through context
4. **Single Responsibility** - Each event does one thing well
5. **Composability** - Events and middleware can be mixed and matched

## Performance

The EventChains pattern is designed for minimal overhead:

- Pipeline built once at construction, cached for reuse
- No decorator overhead on every execution
- Direct method calls through pre-built pipeline
- Suitable for high-iteration scenarios (ML training, batch processing)

## Use Cases

- **Web APIs** - Request/response pipelines
- **Data Processing** - ETL workflows
- **Machine Learning** - Training pipelines with observability
- **Business Workflows** - Order processing, user registration
- **Game AI** - Decision trees, behavior sequences
- **IoT** - Device command processing
- **CI/CD** - Build, test, deploy sequences

## Testing

Events are easy to test in isolation:

```python
def test_validate_input():
    event = ValidateInput()
    context = EventContext({'input': 5})
    result = event.execute(context)
    assert result.success

def test_validate_input_fails_on_negative():
    event = ValidateInput()
    context = EventContext({'input': -1})
    result = event.execute(context)
    assert not result.success
    assert "Invalid input" in result.error
```

## License

MIT License

## Contributing

Contributions welcome! Please ensure:
- Events are stateless
- Middleware doesn't modify event behavior
- No decorators (explicit middleware registration only)
- Code is documented and tested

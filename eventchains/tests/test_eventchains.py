"""
Unit tests for EventChains core components.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eventchains import EventChain, ChainableEvent, EventContext, Result, Middleware, FaultTolerance


# Test Events
class SuccessEvent(ChainableEvent):
    def execute(self, context):
        context.set('executed', True)
        return Result.ok()


class FailureEvent(ChainableEvent):
    def execute(self, context):
        return Result.fail("Intentional failure")


class IncrementEvent(ChainableEvent):
    def execute(self, context):
        current = context.get('counter', 0)
        context.set('counter', current + 1)
        return Result.ok()


# Test Middleware
class CountingMiddleware(Middleware):
    def __init__(self):
        self.call_count = 0
    
    def execute(self, context, next_callable):
        self.call_count += 1
        return next_callable(context)


class LoggingMiddleware(Middleware):
    def __init__(self):
        self.logs = []
    
    def execute(self, context, next_callable):
        event_name = context.get('_current_event', 'Unknown')
        self.logs.append(f"Before {event_name}")
        
        result = next_callable(context)
        
        self.logs.append(f"After {event_name}")
        return result


# Tests
def test_event_context():
    print("Testing EventContext...")
    
    context = EventContext()
    assert context.get('missing') is None
    assert context.get('missing', 'default') == 'default'
    
    context.set('key', 'value')
    assert context.get('key') == 'value'
    assert context.has('key')
    
    context.remove('key')
    assert not context.has('key')
    
    print("  ✓ EventContext tests passed")


def test_result():
    print("Testing Result...")
    
    success = Result.ok()
    assert success.success
    assert success.is_success()
    assert not success.is_failure()
    
    failure = Result.fail("Error message")
    assert not failure.success
    assert failure.is_failure()
    assert not failure.is_success()
    assert failure.error == "Error message"
    
    print("  ✓ Result tests passed")


def test_chainable_event():
    print("Testing ChainableEvent...")
    
    event = SuccessEvent()
    context = EventContext()
    result = event.execute(context)
    
    assert result.success
    assert context.get('executed') == True
    
    print("  ✓ ChainableEvent tests passed")


def test_event_chain_basic():
    print("Testing EventChain basic execution...")
    
    chain = EventChain()
    chain.add_event(IncrementEvent())
    chain.add_event(IncrementEvent())
    chain.add_event(IncrementEvent())
    
    context = EventContext()
    result = chain.execute(context)
    
    assert result.success
    assert context.get('counter') == 3
    
    print("  ✓ EventChain basic execution passed")


def test_event_chain_failure():
    print("Testing EventChain failure handling...")
    
    # STRICT mode - should stop on failure
    chain = EventChain(fault_tolerance=FaultTolerance.STRICT)
    chain.add_event(IncrementEvent())
    chain.add_event(FailureEvent())
    chain.add_event(IncrementEvent())  # Should not execute
    
    context = EventContext()
    result = chain.execute(context)
    
    assert not result.success
    assert context.get('counter') == 1  # Only first event executed
    
    print("  ✓ EventChain failure handling passed")


def test_middleware_execution():
    print("Testing Middleware execution...")
    
    counter = CountingMiddleware()
    
    chain = EventChain()
    chain.add_event(IncrementEvent())
    chain.add_event(IncrementEvent())
    chain.use_middleware(counter)
    
    context = EventContext()
    result = chain.execute(context)
    
    assert result.success
    assert counter.call_count == 2  # Called once per event
    
    print("  ✓ Middleware execution passed")


def test_middleware_order():
    print("Testing Middleware LIFO order...")
    
    logger = LoggingMiddleware()
    
    chain = EventChain()
    chain.add_event(SuccessEvent())
    chain.use_middleware(logger)
    
    context = EventContext()
    result = chain.execute(context)
    
    assert result.success
    assert len(logger.logs) == 2
    assert logger.logs[0].startswith("Before")
    assert logger.logs[1].startswith("After")
    
    print("  ✓ Middleware LIFO order passed")


def test_multiple_middleware():
    print("Testing multiple middleware...")
    
    counter1 = CountingMiddleware()
    counter2 = CountingMiddleware()
    
    chain = EventChain()
    chain.add_event(IncrementEvent())
    chain.add_event(IncrementEvent())
    chain.use_middleware(counter1)
    chain.use_middleware(counter2)
    
    context = EventContext()
    result = chain.execute(context)
    
    assert result.success
    assert counter1.call_count == 2
    assert counter2.call_count == 2
    
    print("  ✓ Multiple middleware passed")


def test_fault_tolerance_lenient():
    print("Testing LENIENT fault tolerance...")
    
    chain = EventChain(fault_tolerance=FaultTolerance.LENIENT)
    chain.add_event(IncrementEvent())
    chain.add_event(FailureEvent())
    chain.add_event(IncrementEvent())  # Should still execute
    
    context = EventContext()
    result = chain.execute(context)
    
    assert result.success  # Overall success despite failure
    assert context.get('counter') == 2  # Both increment events executed
    assert context.has('_error_FailureEvent')  # Error logged in context
    
    print("  ✓ LENIENT fault tolerance passed")


def test_context_data_flow():
    print("Testing context data flow...")
    
    class SetDataEvent(ChainableEvent):
        def execute(self, context):
            context.set('step1', 'data1')
            return Result.ok()
    
    class ReadDataEvent(ChainableEvent):
        def execute(self, context):
            data = context.get('step1')
            context.set('step2', f"processed_{data}")
            return Result.ok()
    
    chain = EventChain()
    chain.add_event(SetDataEvent())
    chain.add_event(ReadDataEvent())
    
    context = EventContext()
    result = chain.execute(context)
    
    assert result.success
    assert context.get('step1') == 'data1'
    assert context.get('step2') == 'processed_data1'
    
    print("  ✓ Context data flow passed")


def run_all_tests():
    print("=" * 60)
    print("Running EventChains Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_event_context,
        test_result,
        test_chainable_event,
        test_event_chain_basic,
        test_event_chain_failure,
        test_middleware_execution,
        test_middleware_order,
        test_multiple_middleware,
        test_fault_tolerance_lenient,
        test_context_data_flow,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ Test failed: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ Test error: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

"""
Simple example demonstrating the EventChains pattern.
"""

from eventchains import EventChain, ChainableEvent, EventContext, Result, Middleware


# Define some simple events
class GreetUser(ChainableEvent):
    def execute(self, context):
        name = context.get('name', 'World')
        greeting = f"Hello, {name}!"
        context.set('greeting', greeting)
        print(f"Event: {greeting}")
        return Result.ok()


class AddTimestamp(ChainableEvent):
    def execute(self, context):
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        context.set('timestamp', timestamp)
        print(f"Event: Added timestamp {timestamp}")
        return Result.ok()


class FormatMessage(ChainableEvent):
    def execute(self, context):
        greeting = context.get('greeting')
        timestamp = context.get('timestamp')
        message = f"{greeting} (at {timestamp})"
        context.set('final_message', message)
        print(f"Event: Formatted message")
        return Result.ok()


# Define middleware
class LoggingMiddleware(Middleware):
    def execute(self, context, next_callable):
        event_name = context.get('_current_event', 'Unknown')
        print(f"[LOG] Starting {event_name}")
        
        result = next_callable(context)
        
        if result.success:
            print(f"[LOG] Completed {event_name}")
        else:
            print(f"[LOG] Failed {event_name}: {result.error}")
        
        return result


class TimingMiddleware(Middleware):
    def execute(self, context, next_callable):
        import time
        event_name = context.get('_current_event', 'Unknown')
        
        start = time.perf_counter()
        result = next_callable(context)
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"[TIMING] {event_name} took {elapsed:.2f}ms")
        
        return result


def main():
    print("=" * 60)
    print("EventChains Simple Example")
    print("=" * 60)
    print()
    
    # Build the chain
    chain = (EventChain()
        .add_event(GreetUser())
        .add_event(AddTimestamp())
        .add_event(FormatMessage())
        .use_middleware(LoggingMiddleware())
        .use_middleware(TimingMiddleware()))
    
    print(f"Chain built: {chain}")
    print()
    
    # Execute the chain
    print("Executing chain...")
    print("-" * 60)
    
    context = EventContext({'name': 'Alice'})
    result = chain.execute(context)
    
    print("-" * 60)
    print()
    
    # Check result
    if result.success:
        print("✓ Chain executed successfully!")
        print(f"Final message: {context.get('final_message')}")
    else:
        print(f"✗ Chain failed: {result.error}")
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()

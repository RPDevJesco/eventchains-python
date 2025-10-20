"""
EventChain - Orchestrates sequential execution of events through middleware.
"""

from .result import Result

class FaultTolerance:
    """Enumeration of fault tolerance modes for event chains."""
    STRICT = "strict"           # Any failure stops the chain
    LENIENT = "lenient"         # Non-critical failures continue
    BEST_EFFORT = "best_effort" # All events attempted regardless of failures
    CUSTOM = "custom"           # User-defined failure handling


class EventChain:
    """
    Orchestrates sequential execution of events through a middleware pipeline.
    
    The chain manages:
    - Sequential event execution
    - Middleware pipeline (LIFO order)
    - Error propagation based on fault tolerance
    - Shared context management
    """
    
    def __init__(self, fault_tolerance=FaultTolerance.STRICT):
        """
        Initialize an EventChain.
        
        Args:
            fault_tolerance: How to handle failures (default: STRICT)
        """
        self._events = []
        self._middleware = []
        self._fault_tolerance = fault_tolerance
        self._pipeline = None
        self._pipeline_built = False
    
    def add_event(self, event):
        """
        Add an event to the chain.
        
        Args:
            event: ChainableEvent instance to add
            
        Returns:
            self (for method chaining)
        """
        self._events.append(event)
        self._pipeline_built = False  # Invalidate cached pipeline
        return self
    
    def use_middleware(self, middleware):
        """
        Add middleware to the chain.
        Middleware executes in LIFO order (reverse of registration).
        
        Args:
            middleware: Middleware instance to add
            
        Returns:
            self (for method chaining)
        """
        self._middleware.append(middleware)
        self._pipeline_built = False  # Invalidate cached pipeline
        return self
    
    def execute(self, context):
        """
        Execute all events in the chain through the middleware pipeline.
        
        Args:
            context: EventContext containing shared state
            
        Returns:
            Result indicating overall success or failure
        """
        # Build pipeline once and cache it
        if not self._pipeline_built:
            self._pipeline = self._build_pipeline()
            self._pipeline_built = True
        
        # Execute all events through the pipeline
        for event in self._events:
            # Store current event name in context for middleware
            context.set('_current_event', event.__class__.__name__)
            
            result = self._pipeline(event, context)
            
            # Handle failure based on fault tolerance
            if not result.success:
                if self._fault_tolerance == FaultTolerance.STRICT:
                    return result
                elif self._fault_tolerance == FaultTolerance.LENIENT:
                    # Log but continue (could add to context)
                    context.set(f'_error_{event.__class__.__name__}', result.error)
                    continue
                elif self._fault_tolerance == FaultTolerance.BEST_EFFORT:
                    # Always continue
                    context.set(f'_error_{event.__class__.__name__}', result.error)
                    continue
                # CUSTOM mode would require additional configuration
        
        return Result.ok()
    
    def _build_pipeline(self):
        """
        Build the middleware pipeline.
        Middleware wraps in LIFO order (reverse of registration).
        
        Returns:
            Function that executes an event through all middleware
        """
        def execute_event(event, context):
            """Inner function that actually executes the event."""
            return event.execute(context)
        
        # Start with the base executor
        pipeline = execute_event
        
        # Wrap in middleware (reverse order for LIFO)
        for middleware in reversed(self._middleware):
            pipeline = self._create_middleware_wrapper(middleware, pipeline)
        
        return pipeline
    
    def _create_middleware_wrapper(self, middleware, next_pipeline):
        """
        Create a wrapper that calls middleware with the next pipeline.
        
        Args:
            middleware: Middleware instance to wrap
            next_pipeline: The next function in the pipeline
            
        Returns:
            Function that executes the middleware
        """
        def wrapper(event, context):
            return middleware.execute(
                context,
                lambda ctx: next_pipeline(event, ctx)
            )
        return wrapper
    
    def clear_events(self):
        """Remove all events from the chain."""
        self._events.clear()
        self._pipeline_built = False
        return self
    
    def clear_middleware(self):
        """Remove all middleware from the chain."""
        self._middleware.clear()
        self._pipeline_built = False
        return self
    
    def reset(self):
        """Clear both events and middleware."""
        self.clear_events()
        self.clear_middleware()
        return self
    
    def event_count(self):
        """Return the number of events in the chain."""
        return len(self._events)
    
    def middleware_count(self):
        """Return the number of middleware in the chain."""
        return len(self._middleware)
    
    def __repr__(self):
        return (f"EventChain(events={len(self._events)}, "
                f"middleware={len(self._middleware)}, "
                f"fault_tolerance={self._fault_tolerance})")

"""
Middleware - Base class for cross-cutting concerns that wrap event execution.
"""

class Middleware:
    """
    Base class for middleware that wraps event execution.
    
    Middleware provides cross-cutting concerns like logging, timing, error handling, etc.
    Middleware executes in LIFO order (reverse of registration) - like gift wrapping.
    """
    
    def execute(self, context, next_callable):
        """
        Execute the middleware logic.
        
        Args:
            context: EventContext containing shared state
            next_callable: Function to call to continue the chain (must be called)
            
        Returns:
            Result from the next callable (or modified result)
            
        Example:
            def execute(self, context, next_callable):
                # Before logic
                print("Before event")
                
                # Call next in chain
                result = next_callable(context)
                
                # After logic
                print("After event")
                
                return result
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement execute()")
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"
    
    def __str__(self):
        return self.__class__.__name__

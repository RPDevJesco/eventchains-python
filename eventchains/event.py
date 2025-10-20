"""
ChainableEvent - Base class for discrete units of business logic in an event chain.
"""

from .result import Result

class ChainableEvent:
    """
    Base class for events in an event chain.
    Each event represents a discrete unit of business logic.
    
    Events should be stateless - all state flows through the EventContext.
    """
    
    def execute(self, context):
        """
        Execute the event logic.
        
        Args:
            context: EventContext containing shared state
            
        Returns:
            Result indicating success or failure
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement execute()")
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"
    
    def __str__(self):
        return self.__class__.__name__

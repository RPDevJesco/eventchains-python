"""
EventContext - Shared data container that flows through the event chain.
"""

class EventContext:
    """
    A shared data container (dictionary/key-value store) that flows through the entire chain.
    Enables communication between sequential events.
    """
    
    def __init__(self, data=None):
        """
        Initialize the EventContext with optional initial data.
        
        Args:
            data: Dictionary of initial context data (optional)
        """
        self._data = data if data is not None else {}
    
    def get(self, key, default=None):
        """
        Get a value from the context.
        
        Args:
            key: The key to retrieve
            default: Default value if key not found
            
        Returns:
            The value associated with the key, or default if not found
        """
        return self._data.get(key, default)
    
    def set(self, key, value):
        """
        Set a value in the context.
        
        Args:
            key: The key to set
            value: The value to store
            
        Returns:
            self (for method chaining)
        """
        self._data[key] = value
        return self
    
    def has(self, key):
        """
        Check if a key exists in the context.
        
        Args:
            key: The key to check
            
        Returns:
            True if key exists, False otherwise
        """
        return key in self._data
    
    def remove(self, key):
        """
        Remove a key from the context.
        
        Args:
            key: The key to remove
            
        Returns:
            self (for method chaining)
        """
        if key in self._data:
            del self._data[key]
        return self
    
    def clear(self):
        """Clear all data from the context."""
        self._data.clear()
        return self
    
    def keys(self):
        """Return all keys in the context."""
        return self._data.keys()
    
    def values(self):
        """Return all values in the context."""
        return self._data.values()
    
    def items(self):
        """Return all key-value pairs in the context."""
        return self._data.items()
    
    def to_dict(self):
        """Return a copy of the internal data dictionary."""
        return self._data.copy()
    
    def __repr__(self):
        return f"EventContext({self._data})"
    
    def __str__(self):
        return str(self._data)

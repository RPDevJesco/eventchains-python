"""
Result - Represents the outcome of an event execution.
"""

class Result:
    """
    Represents the outcome of an event execution.
    Contains success status and optional error information.
    """
    
    def __init__(self, success, error=None, data=None):
        """
        Initialize a Result.
        
        Args:
            success: Boolean indicating if the event succeeded
            error: Optional error message or exception
            data: Optional additional data about the result
        """
        self.success = success
        self.error = error
        self.data = data
    
    @staticmethod
    def ok(data=None):
        """
        Create a successful result.
        
        Args:
            data: Optional data to include with the success result
            
        Returns:
            Result instance indicating success
        """
        return Result(True, data=data)
    
    @staticmethod
    def fail(error, data=None):
        """
        Create a failed result.
        
        Args:
            error: Error message or exception
            data: Optional additional data about the failure
            
        Returns:
            Result instance indicating failure
        """
        return Result(False, error=error, data=data)
    
    def is_success(self):
        """Return True if the result indicates success."""
        return self.success
    
    def is_failure(self):
        """Return True if the result indicates failure."""
        return not self.success
    
    def __bool__(self):
        """Allow Result to be used in boolean context (if result: ...)"""
        return self.success
    
    def __repr__(self):
        if self.success:
            return f"Result.ok(data={self.data})"
        else:
            return f"Result.fail(error={self.error}, data={self.data})"
    
    def __str__(self):
        if self.success:
            return "Success"
        else:
            return f"Failure: {self.error}"

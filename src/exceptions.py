class ProjectException(Exception):
    """
    Base exception for the project.
    Wraps an optional original exception for debugging.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.message = message
        self.original_exception = original_exception

    def __str__(self):
        if self.original_exception:
            return f"{self.message}\nCaused by: {repr(self.original_exception)}"
        return self.message
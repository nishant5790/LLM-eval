class LLMProfboxError(Exception):
    """Base exception for LLMprofbox package"""
    pass

class MetricCalculationError(LLMProfboxError):
    """Raised when there's an error in metric calculation"""
    pass

class BedrockEvaluationError(LLMProfboxError):
    """Raised when there's an error with Bedrock model evaluation"""
    pass

class ModelNotSupportedError(LLMProfboxError):
    """Raised when an unsupported model is selected"""
    pass

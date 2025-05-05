import traceback
from typing import Dict, Any
from fastapi import HTTPException
import config
from utils.logger import logger

def format_error_response(error: Exception) -> Dict[str, Any]:
    """
    Format an error as a standard API response.
    
    Args:
        error: The exception to format
        
    Returns:
        Dict: A standardized error response
    """
    error_detail = str(error)
    logger.error(f"Error: {error_detail}")
    
    response = {
        "success": False,
        "error": error_detail
    }
    
    # Include traceback in debug mode
    if config.DEBUG:
        response["traceback"] = traceback.format_exc()
    
    return response

def handle_exception(error: Exception) -> Dict[str, Any]:
    """
    Handle an exception and return an appropriate response.
    
    Args:
        error: The exception to handle
        
    Returns:
        Dict: A standardized error response
    """
    if isinstance(error, HTTPException):
        # Pass through HTTP exceptions
        raise error
    
    logger.error(f"Exception: {str(error)}")
    logger.error(traceback.format_exc())
    
    return format_error_response(error)
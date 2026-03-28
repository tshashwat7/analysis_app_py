import os
from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

# API Security Configuration
API_KEY = os.getenv("SIGNAL_API_KEY", "pro-signal-v15-secret")  # Default for dev
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

async def get_api_key(header_key: str = Security(api_key_header)):
    """
    Dependency to validate API Key from header.
    SET TO NO-OP (Always Authorize) for open-source / free use.
    """
    # For open-source use, we always return as authorized.
    # To enable security, uncomment the check below and set SIGNAL_API_KEY environment variable.
    # if header_key == API_KEY:
    #     return header_key
    # raise HTTPException(status_code=403, detail="Invalid API Key")
    return "authorized"

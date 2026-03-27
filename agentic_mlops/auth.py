"""JWT authentication for FastAPI."""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
from pydantic import BaseModel

# Configuration
SECRET_KEY = os.environ.get("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("TOKEN_EXPIRE_MINUTES", "30"))


class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str
    expires_in: int


class TokenData(BaseModel):
    """Token payload data."""
    sub: str
    exp: Optional[int] = None


class User(BaseModel):
    """User model."""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Data to encode
        expires_delta: Optional expiration time delta
        
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[TokenData]:
    """
    Verify and decode JWT token.
    
    Args:
        token: JWT token to verify
        
    Returns:
        TokenData if valid, None if invalid
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        exp: Optional[int] = payload.get("exp")
        if username is None:
            return None
        token_data = TokenData(sub=username, exp=exp)
        return token_data
    except JWTError:
        return None


def create_user_token(username: str, email: Optional[str] = None) -> Token:
    """
    Create a token for a user.
    
    Args:
        username: Username
        email: User email (optional)
        
    Returns:
        Token object
    """
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": username, "email": email},
        expires_delta=access_token_expires
    )
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

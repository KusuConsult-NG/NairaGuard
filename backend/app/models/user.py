from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr


class UserBase(BaseModel):
    """Base user model"""

    email: EmailStr
    username: str
    full_name: str


class UserCreate(UserBase):
    """User creation model"""

    password: str


class UserLogin(BaseModel):
    """User login model"""

    username: str
    password: str


class User(UserBase):
    """User model"""

    id: str
    is_active: bool = True
    created_at: str
    last_login: Optional[str] = None


class UserUpdate(BaseModel):
    """User update model"""

    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = None


class Token(BaseModel):
    """Token model"""

    access_token: str
    token_type: str
    expires_in: int


class TokenData(BaseModel):
    """Token data model"""

    username: Optional[str] = None

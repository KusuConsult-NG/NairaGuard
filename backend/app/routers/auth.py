from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime, timedelta
from typing import Optional
import jwt
from passlib.context import CryptContext

from backend.app.core.config import settings
from backend.app.models.user import User, UserCreate, UserLogin, Token

router = APIRouter()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def verify_token(token: str = Depends(oauth2_scheme)):
    """Verify and decode a JWT token"""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.post("/register", response_model=User)
async def register_user(user_data: UserCreate):
    """
    Register a new user
    """
    try:
        # In a real application, you would:
        # 1. Check if user already exists
        # 2. Hash the password
        # 3. Save to database
        # 4. Return user data
        
        # For now, return mock data
        hashed_password = get_password_hash(user_data.password)
        
        user = User(
            id="mock-user-id",
            email=user_data.email,
            username=user_data.username,
            full_name=user_data.full_name,
            is_active=True,
            created_at=datetime.utcnow().isoformat()
        )
        
        return user
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Registration failed: {str(e)}")

@router.post("/login", response_model=Token)
async def login_user(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login user and return access token
    """
    try:
        # In a real application, you would:
        # 1. Verify user credentials against database
        # 2. Check if user is active
        # 3. Generate and return token
        
        # For now, accept any username/password for demo
        if not form_data.username or not form_data.password:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Mock user authentication
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": form_data.username}, expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.get("/me", response_model=User)
async def get_current_user(current_user_id: str = Depends(verify_token)):
    """
    Get current user information
    """
    try:
        # In a real application, you would fetch user from database
        # For now, return mock data
        user = User(
            id=current_user_id,
            email="user@example.com",
            username=current_user_id,
            full_name="Demo User",
            is_active=True,
            created_at=datetime.utcnow().isoformat()
        )
        
        return user
        
    except Exception as e:
        raise HTTPException(status_code=404, detail="User not found")

@router.post("/logout")
async def logout_user(current_user_id: str = Depends(verify_token)):
    """
    Logout user (invalidate token)
    """
    # In a real application, you would:
    # 1. Add token to blacklist
    # 2. Remove from active sessions
    # 3. Log the logout event
    
    return {"message": "Successfully logged out"}

@router.get("/verify-token")
async def verify_token_endpoint(current_user_id: str = Depends(verify_token)):
    """
    Verify if a token is valid
    """
    return {
        "valid": True,
        "user_id": current_user_id,
        "message": "Token is valid"
    }

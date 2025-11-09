from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.services import llm_connector
from api.utils.auth_utils import hash_password, verify_password
from api.services import ad_service
from api.services import seo_service

router = APIRouter(prefix="/auth", tags=["Auth"])

# simple in-memory store; replace with database lookups in production
USERS = {}

class UserLogin(BaseModel):
    email: str
    password: str

class UserRegister(BaseModel):
    email: str
    password: str

@router.post("/register")
def register_user(user: UserRegister):
    if user.email in USERS:
        raise HTTPException(status_code=400, detail="User already exists")
    USERS[user.email] = hash_password(user.password)
    return {"message": "User registered successfully"}

@router.post("/login")
def login_user(user: UserLogin):
    if user.email not in USERS or not verify_password(user.password, USERS[user.email]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"message": "Login successful"}

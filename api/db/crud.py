from sqlalchemy.orm import Session
from .models import Strategy

def save_strategy(db: Session, business_name: str, industry: str, goal: str, strategy_text: str):
    new_strategy = Strategy(
        business_name=business_name,
        industry=industry,
        goal=goal,
        strategy_text=strategy_text
    )
    db.add(new_strategy)
    db.commit()
    db.refresh(new_strategy)
    return new_strategy

def get_strategies(db: Session, skip: int = 0, limit: int = 10):
    return db.query(Strategy).order_by(Strategy.id.desc()).offset(skip).limit(limit).all()

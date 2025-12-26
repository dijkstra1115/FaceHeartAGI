from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime,
    UniqueConstraint
)
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./faceheart.db"

# 建立 engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    pool_size=10,  # 連接池大小
    max_overflow=20,  # 超過pool_size後最多創建的連接數
    pool_pre_ping=True,  # 使用前檢查連接是否有效
    pool_recycle=3600  # 1小時後回收連接
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base model
Base = declarative_base()

class ConversationTurn(Base):
    __tablename__ = "conversation_turns"
    __table_args__ = (
        UniqueConstraint("device_id", "turn_number", name="uix_device_turn"),
    )

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, index=True, nullable=False)
    turn_number = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_intent = Column(Text)
    system_response = Column(Text)
    fhir_data = Column(Text, nullable=True)

class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, index=True, nullable=False)
    summary_index = Column(Integer, nullable=False)
    content = Column(Text)

def init_db():
    """
    建立所有資料表（如果尚未存在）。
    """
    Base.metadata.create_all(bind=engine)
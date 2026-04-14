from sqlalchemy import Column, String, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from database import Base
from datetime import datetime, timedelta
import uuid

# Helper for IST (Indian Standard Time)
def get_ist():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    session_id     = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id    = Column(String(50), nullable=False)
    created_at     = Column(DateTime, default=get_ist)
    last_active_at = Column(DateTime, default=get_ist)

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    message_id  = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id  = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.session_id"))
    employee_id = Column(String(50), nullable=False)
    timestamp   = Column(DateTime, default=get_ist)
    query       = Column(Text, nullable=False)
    answer      = Column(Text, nullable=False)
    rating      = Column(String(20), nullable=True)

class EmployeeDevice(Base):
    __tablename__ = "employee_devices"

    device_id   = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id = Column(String(50), nullable=False)
    platform    = Column(String(10), nullable=True)
    push_token  = Column(Text, nullable=True)
    app_version = Column(String(20), nullable=True)
    registered_at = Column(DateTime, default=get_ist)
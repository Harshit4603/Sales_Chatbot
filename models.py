from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID
from database import Base
from datetime import datetime, timedelta
import uuid

# Helper for IST (Indian Standard Time)
def get_ist():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)

class Employee(Base):
    __tablename__ = "employees"

    employee_id   = Column(String(50), primary_key=True)
    name          = Column(String(100), nullable=False)
    email         = Column(String(100), unique=True, nullable=False)
    password_hash = Column(Text, nullable=False)
    role          = Column(String(10), nullable=False)
    created_at    = Column(DateTime, default=get_ist)

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
    used_internet = Column(String(5), nullable=True)
    used_internet = Column(String(5), nullable=True)
    is_saved    = Column(Boolean, default=False)

class IngestionLog(Base):
    __tablename__ = "ingestion_logs"

    log_id        = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename      = Column(String(255), nullable=False)
    file_type     = Column(String(10),  nullable=False)   # pdf / docx / pptx
    doc_category  = Column(String(50),  nullable=True)    # product / sop / policy …
    sections      = Column(String(20),  nullable=True)    # sections extracted
    chunks        = Column(String(20),  nullable=True)    # vectors upserted
    status        = Column(String(20),  nullable=False, default="success")  # success / failed
    error_detail  = Column(Text,        nullable=True)    # error message if failed
    uploaded_by   = Column(String(50),  nullable=True)    # employee_id of uploader
    uploaded_at   = Column(DateTime,    default=get_ist)
    replaced_previous = Column(Boolean, default=False)    # True if old vectors were deleted first


class EmployeeDevice(Base):
    __tablename__ = "employee_devices"

    device_id   = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id = Column(String(50), nullable=False)
    platform    = Column(String(10), nullable=True)
    push_token  = Column(Text, nullable=True)
    app_version = Column(String(20), nullable=True)
    registered_at = Column(DateTime, default=get_ist)

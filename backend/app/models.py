from time import timezone
import uuid
import datetime
from sqlalchemy.sql import func
from sqlalchemy import String, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from app.db import Base

class Document(Base):
    __tablename__ = "documents"

    id:Mapped[uuid.UUID] = mapped_column(primary_key = True,default=uuid.uuid4  )
    filename:Mapped[str] = mapped_column(String,nullable =False)
    status:Mapped[str] = mapped_column(String,nullable=False, default="pending")
    created_at:Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True),default=func.now(), nullable =False )
    updated_at:Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True),default=func.now(),onupdate=func.now(), nullable =False )
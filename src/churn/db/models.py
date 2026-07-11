"""SQLAlchemy ORM model for the customers table."""

from __future__ import annotations

from sqlalchemy import Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"

    customer_id: Mapped[str] = mapped_column(String, primary_key=True)
    tenure_months: Mapped[int] = mapped_column(Integer)
    contract_type: Mapped[str] = mapped_column(String)
    internet_service: Mapped[str] = mapped_column(String)
    payment_method: Mapped[str] = mapped_column(String)
    monthly_charges: Mapped[float] = mapped_column(Float)
    total_charges: Mapped[float] = mapped_column(Float)
    support_calls: Mapped[int] = mapped_column(Integer)
    churn: Mapped[int] = mapped_column(Integer)

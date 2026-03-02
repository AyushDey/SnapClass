from db import Base
from sqlalchemy import String, Integer, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector

class BookletItem(Base):
    __tablename__= 'booklet_items'
    id: Mapped[int] = mapped_column(Integer, primary_key=True, unique=True, autoincrement='auto')
    item_name: Mapped[str] = mapped_column(String)
    category_id: Mapped[int] = mapped_column(Integer, ForeignKey('booklet_category.id'))
    image_hash: Mapped[str] = mapped_column(String, unique=True, index=True)
    embedding = mapped_column(Vector(512))

class BookletCategory(Base):
    __tablename__ = 'booklet_category'
    id: Mapped[int] = mapped_column(Integer, primary_key=True, unique=True, autoincrement='auto')
    category_name: Mapped[str] = mapped_column(String)
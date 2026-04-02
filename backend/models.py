from db import Base
from sqlalchemy import String, Integer, ForeignKey, Column
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector


class BookletItem(Base):
    __tablename__ = "booklet_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, unique=True, autoincrement="auto")
    item_name: Mapped[str] = mapped_column(String, unique=True, index=True)
    embeddings: Mapped[list["BookletEmbedding"]] = relationship(back_populates="item")


class BookletCategory(Base):
    __tablename__ = "booklet_category"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, unique=True, autoincrement="auto")
    category_name: Mapped[str] = mapped_column(String)
    embeddings: Mapped[list["BookletEmbedding"]] = relationship(back_populates="category")


class BookletEmbedding(Base):
    __tablename__ = "booklet_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, unique=True, autoincrement="auto")
    booklet_item_id: Mapped[int] = mapped_column(Integer, ForeignKey("booklet_items.id"))
    booklet_category_id: Mapped[int] = mapped_column(Integer, ForeignKey("booklet_category.id"))
    image_hash: Mapped[str] = mapped_column(String, unique=True, index=True)
    embedding = mapped_column(Vector(512))

    item: Mapped[BookletItem] = relationship(back_populates="embeddings")
    category: Mapped[BookletCategory] = relationship(back_populates="embeddings")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
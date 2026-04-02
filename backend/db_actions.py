from sqlalchemy.orm import Session, joinedload
from sqlalchemy import select, delete
from sqlalchemy.dialects.postgresql import insert
from models import BookletItem, BookletCategory, BookletEmbedding, User
from db import sessionLocal

class DBActions:
    """Encapsulates all database interactions for the classifier."""
    def __init__(self, session: Session):
        self.session = session

    def get_existing_hashes(self) -> set:
        """Fetch set of all image hashes currently in the DB."""
        stmt = select(BookletEmbedding.image_hash)
        return {row[0] for row in self.session.execute(stmt).fetchall()}

    def get_or_create_item(self, item_name: str) -> int:
        """Finds an item by name or creates it, returning the ID."""
        stmt = select(BookletItem).where(BookletItem.item_name == item_name)
        item = self.session.scalars(stmt).first()

        if item:
            return item.id

        new_item = BookletItem(item_name=item_name)
        self.session.add(new_item)
        self.session.flush()
        return new_item.id

    def get_or_create_category(self, category_name: str | None) -> int:
        """Finds a category by name or creates it, returning the ID."""
        if not category_name:
            category_name = "Uncategorized"
            
        stmt = select(BookletCategory).where(BookletCategory.category_name == category_name)
        category = self.session.scalars(stmt).first()
        
        if category:
            return category.id
            
        new_category = BookletCategory(category_name=category_name)
        self.session.add(new_category)
        self.session.flush() # Flush to populate the new ID immediately
        return new_category.id

    def insert_embeddings(self, embeddings: list):
        """Bulk insert embeddings, ignoring duplicates."""
        if not embeddings:
            return
        stmt = insert(BookletEmbedding).values(embeddings)
        stmt = stmt.on_conflict_do_nothing(index_elements=["image_hash"])
        self.session.execute(stmt)

    def get_all_embeddings(self):
        """Fetch all booklet embeddings with their linked items."""
        stmt = select(BookletEmbedding).options(joinedload(BookletEmbedding.item))
        return self.session.scalars(stmt).all()

    def delete_embeddings(self, embedding_ids: list):
        """Delete embeddings by ID."""
        if not embedding_ids:
            return
        self.session.execute(delete(BookletEmbedding).where(BookletEmbedding.id.in_(embedding_ids)))

    def commit(self):
        self.session.commit()

    def get_category_by_id(self, category_id: int) -> str:
        """
        Get category name from id
        """
        category = self.session.get(BookletCategory, category_id)

        return category.category_name if category else 'Unkown'

    def get_user_by_email(self, email: str):
        stmt = select(User).where(User.email == email)
        return self.session.scalars(stmt).first()

    def create_user(self, email: str, password: str):
        user = User(email=email, password=password)
        self.session.add(user)
        self.session.flush()  
        return user
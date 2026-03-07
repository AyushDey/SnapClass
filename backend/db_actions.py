from sqlalchemy.orm import Session
from sqlalchemy import select, delete
from sqlalchemy.dialects.postgresql import insert
from models import BookletItem, BookletCategory

class DBActions:
    """Encapsulates all database interactions for the classifier."""
    def __init__(self, session: Session):
        self.session = session

    def get_existing_hashes(self) -> set:
        """Fetch set of all image hashes currently in the DB."""
        stmt = select(BookletItem.image_hash)
        return {row[0] for row in self.session.execute(stmt).fetchall()}

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

    def insert_items(self, items: list):
        """Bulk insert items, ignoring duplicates."""
        if not items:
            return
        stmt = insert(BookletItem).values(items)
        stmt = stmt.on_conflict_do_nothing(index_elements=['image_hash'])
        self.session.execute(stmt)

    def get_all_items(self):
        """Fetch all booklet items."""
        stmt = select(BookletItem)
        return self.session.scalars(stmt).all()

    def delete_items(self, item_ids: list):
        """Delete items by ID."""
        if not item_ids:
            return
        self.session.execute(delete(BookletItem).where(BookletItem.id.in_(item_ids)))

    def commit(self):
        self.session.commit()

    def get_category_by_id(self, category_id: int) -> str:
        """
        Get category name from id
        """
        category = self.session.get(BookletCategory, category_id)

        return category.category_name if category else 'Unkown'

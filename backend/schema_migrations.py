from sqlalchemy import Column, ForeignKey, Integer, MetaData, String, Table, inspect, insert, select, text
from pgvector.sqlalchemy import Vector

from models import Base


def initialize_database(engine) -> None:
    """Create the normalized schema, migrating legacy booklet_items data if needed."""
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())

    if _needs_booklet_migration(inspector, table_names):
        _migrate_legacy_booklet_items(engine)

    Base.metadata.create_all(bind=engine)


def _needs_booklet_migration(inspector, table_names: set[str]) -> bool:
    if "booklet_items" not in table_names:
        return False

    column_names = {column["name"] for column in inspector.get_columns("booklet_items")}
    legacy_columns = {"category_id", "image_hash", "embedding"}
    return legacy_columns.issubset(column_names) and "booklet_embeddings" not in table_names


def _migrate_legacy_booklet_items(engine) -> None:
    migration_meta = MetaData()
    legacy_meta = MetaData()

    legacy_items = Table("booklet_items", legacy_meta, autoload_with=engine)
    Table("booklet_category", migration_meta, autoload_with=engine)
    booklet_items_new = Table(
        "booklet_items_new",
        migration_meta,
        Column("id", Integer, primary_key=True),
        Column("item_name", String, nullable=False, unique=True, index=True),
    )
    booklet_embeddings = Table(
        "booklet_embeddings",
        migration_meta,
        Column("id", Integer, primary_key=True),
        Column("booklet_item_id", Integer, ForeignKey("booklet_items_new.id"), nullable=False),
        Column("booklet_category_id", Integer, ForeignKey("booklet_category.id"), nullable=False),
        Column("image_hash", String, nullable=False, unique=True, index=True),
        Column("embedding", Vector(512), nullable=False),
    )

    with engine.begin() as conn:
        migration_meta.create_all(conn)

        item_names = [
            {"item_name": item_name}
            for item_name in conn.execute(select(legacy_items.c.item_name).distinct()).scalars()
        ]
        if item_names:
            conn.execute(insert(booklet_items_new), item_names)

        item_id_map = {
            item_name: item_id
            for item_id, item_name in conn.execute(
                select(booklet_items_new.c.id, booklet_items_new.c.item_name)
            )
        }

        embedding_rows = []
        for row in conn.execute(
            select(
                legacy_items.c.item_name,
                legacy_items.c.category_id,
                legacy_items.c.image_hash,
                legacy_items.c.embedding,
            )
        ):
            embedding_rows.append(
                {
                    "booklet_item_id": item_id_map[row.item_name],
                    "booklet_category_id": row.category_id,
                    "image_hash": row.image_hash,
                    "embedding": row.embedding,
                }
            )

        if embedding_rows:
            conn.execute(insert(booklet_embeddings), embedding_rows)

        legacy_items.drop(conn)
        conn.execute(text("ALTER TABLE booklet_items_new RENAME TO booklet_items"))

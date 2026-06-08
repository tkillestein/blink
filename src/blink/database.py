from datetime import datetime

from loguru import logger
from peewee import DateTimeField, Model, PostgresqlDatabase, PrimaryKeyField
from pgvector.peewee import VectorField

from blink.config import BlinkSettings
from blink.utils import module_rng

settings = BlinkSettings()

db = PostgresqlDatabase(
    "embeddingverse",
    user=settings.postgres_user,
    password=settings.postgres_password.get_secret_value()
    if settings.postgres_password
    else None,
    host=settings.postgres_host,
    port=settings.postgres_port,
)


class BaseModel(Model):
    created_at = DateTimeField(default=datetime.now)

    class Meta:
        database = db


class StampEmbedding(BaseModel):
    id = PrimaryKeyField()
    embedding = VectorField()


def init_db() -> None:
    logger.debug("Initialising database")
    with db:
        db.execute_sql("CREATE EXTENSION IF NOT EXISTS vector;")
        db.create_tables([StampEmbedding], safe=True)
        StampEmbedding.add_index("embedding vector_l2_ops", using="hnsw")


def test_vectordb() -> None:
    init_db()
    logger.info("Seeding database with random vectors, have fuuuuuuun")

    with db:
        for _ in range(100):
            StampEmbedding.create(embedding=module_rng.normal(size=(128,)))

        reference_vector = StampEmbedding.get(1)

        qs = (
            StampEmbedding.select()
            .filter(StampEmbedding.id != reference_vector.id)
            .order_by(StampEmbedding.embedding.l2_distance(reference_vector.embedding))
            .limit(5)
        )

        q_ids = [q.id for q in qs]
        logger.info(f"Retrieved vectors: {q_ids}")

        logger.info("Tearing down")
        StampEmbedding.delete().execute()


if __name__ == "__main__":
    test_vectordb()

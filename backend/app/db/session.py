from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.database_url,
    echo=(settings.environment == "development"),
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def init_db():
    """Create all tables defined via SQLAlchemy models."""
    # Import all models so Base.metadata is populated
    from app.db import models  # noqa: F401
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
    # Ensure ANON_USER exists for prototype
    async with AsyncSessionLocal() as session:
        from sqlalchemy import select
        import uuid
        anon_uuid = uuid.UUID('00000000-0000-0000-0000-000000000001')
        anon = (await session.execute(select(models.User).where(models.User.id == anon_uuid))).scalars().first()
        if not anon:
            session.add(models.User(
                id=anon_uuid, 
                email="anon@localhost", 
                display_name="Anonymous", 
                hashed_password="noop"
            ))
            await session.commit()


async def get_db():
    """FastAPI dependency that yields a DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

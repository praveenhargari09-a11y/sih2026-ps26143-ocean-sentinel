import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool
from loguru import logger
from backend.db.models import Base

DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite+aiosqlite:///./data/oil_spill.db')

async_engine = create_async_engine(
    DATABASE_URL,
    connect_args={'check_same_thread': False} if 'sqlite' in DATABASE_URL else {},
    poolclass=StaticPool if 'sqlite' in DATABASE_URL else None,
    echo=False,
)

AsyncSessionLocal = async_sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

async def init_db():
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info('Database tables initialised')

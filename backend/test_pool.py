import asyncio

from db.pool import create_pool, close_pool


async def main():
    pool = await create_pool()

    async with pool.acquire() as conn:
        version = await conn.fetchval("SELECT version();")
        print(version)

    await close_pool()


asyncio.run(main())
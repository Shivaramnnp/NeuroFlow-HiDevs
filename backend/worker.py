import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("worker")


async def main():
    logger.info("Starting NeuroFlow background worker...")
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())

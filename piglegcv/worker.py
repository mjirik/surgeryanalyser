import os

import redis
from rq import Connection, Queue, Worker
from loguru import logger

listen = ["default"]

redis_url = os.getenv("REDISTOGO_URL", "redis://localhost:6379")

conn = redis.from_url(redis_url)

if __name__ == "__main__":
    logger.info("Starting worker")
    worker = Worker(list(map(Queue, listen)), connection=conn)
    worker.work()
    # deprecated
    # with Connection(conn):
    #     logger.info("Starting worker")
    #     worker = Worker(list(map(Queue, listen)))
    #     worker.work()

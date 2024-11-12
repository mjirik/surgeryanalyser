import os

import redis
from rq import Queue, Worker
from rq import Connection

from loguru import logger

listen = ["default"]

redis_url = os.getenv("REDISTOGO_URL", "redis://localhost:6379")

conn = redis.from_url(redis_url)

if __name__ == "__main__":
    # logger.info("Starting worker")
    # queues = [Queue(name, connection=conn) for name in listen]
    # worker = Worker(queues, connection=conn)
    # worker.work()
    # deprecated in 2.0.0
    with Connection(conn):
        logger.info("Starting worker")
        worker = Worker(list(map(Queue, listen)))
        worker.work()

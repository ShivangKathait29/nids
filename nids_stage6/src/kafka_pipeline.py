"""
kafka_pipeline.py
==================
Stage 6 - Component 4 (STRETCH GOAL): Kafka-based Streaming Ingestion

Implements the Tier 2 stretch goal from the project plan: a Kafka producer
publishes flow-feature records as they're aggregated, and a Kafka consumer
performs inference on each message — decoupling capture/aggregation from
inference, and giving you a real, industry-standard streaming architecture
to describe in your report ("Kafka-based real-time ingestion pipeline").

Two backends, same pattern as the rest of Stage 6:

    1. KafkaFlowProducer / KafkaFlowConsumer (REAL) — uses `kafka-python`
       against an actual running Kafka broker. Requires:
           pip install kafka-python
           A running Kafka broker (e.g. `docker run -p 9092:9092 apache/kafka`)
       NOT available in this dev environment — no network access to
       install kafka-python or run a broker.

    2. InMemoryQueueProducer / InMemoryQueueConsumer (tested) — a drop-in
       replacement using Python's `queue.Queue`, matching the exact same
       interface. Lets the full producer -> consumer -> inference flow be
       built and tested end-to-end without a real broker. Swapping to the
       real Kafka classes requires changing ONLY the import/instantiation
       — no other code changes, since both expose `.send(record)` and
       `.poll()` / iteration.
"""

import json
import queue
import time


# ---------------------------------------------------------------------------
# REAL path: kafka-python against an actual broker
# ---------------------------------------------------------------------------
class KafkaFlowProducer:
    """
    REAL Kafka producer. Requires `pip install kafka-python` and a running
    broker — not available in this dev environment.
    """

    def __init__(self, bootstrap_servers: str = "localhost:9092", topic: str = "nids-flows"):
        try:
            from kafka import KafkaProducer
        except ImportError:
            raise RuntimeError(
                "kafka-python is not installed. Run `pip install kafka-python` and ensure "
                "a Kafka broker is reachable at the given bootstrap_servers. This backend "
                "cannot run in the current dev environment — use InMemoryQueueProducer for "
                "development/testing."
            )
        self.topic = topic
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

    def send(self, record: dict):
        self.producer.send(self.topic, value=record)

    def flush(self):
        self.producer.flush()


class KafkaFlowConsumer:
    """REAL Kafka consumer counterpart to KafkaFlowProducer."""

    def __init__(self, bootstrap_servers: str = "localhost:9092", topic: str = "nids-flows",
                 group_id: str = "nids-inference"):
        try:
            from kafka import KafkaConsumer
        except ImportError:
            raise RuntimeError(
                "kafka-python is not installed. See KafkaFlowProducer's docstring for setup."
            )
        self.consumer = KafkaConsumer(
            topic, bootstrap_servers=bootstrap_servers, group_id=group_id,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="latest",
        )

    def __iter__(self):
        for message in self.consumer:
            yield message.value


# ---------------------------------------------------------------------------
# TESTED path: in-memory queue, same interface
# ---------------------------------------------------------------------------
class InMemoryQueueProducer:
    """
    Dependency-free stand-in for KafkaFlowProducer, backed by a shared
    `queue.Queue`. Same `.send()` interface — code written against this
    class needs zero changes to swap to real Kafka later, aside from
    construction.
    """

    def __init__(self, shared_queue: "queue.Queue" = None):
        self.queue = shared_queue if shared_queue is not None else queue.Queue()

    def send(self, record: dict):
        self.queue.put(record)

    def flush(self):
        pass  # no-op — in-memory queue has no buffering to flush


class InMemoryQueueConsumer:
    """Dependency-free stand-in for KafkaFlowConsumer, backed by the same shared queue."""

    def __init__(self, shared_queue: "queue.Queue", poll_timeout: float = 1.0, max_idle_polls: int = 3):
        self.queue = shared_queue
        self.poll_timeout = poll_timeout
        self.max_idle_polls = max_idle_polls

    def __iter__(self):
        idle_polls = 0
        while idle_polls < self.max_idle_polls:
            try:
                record = self.queue.get(timeout=self.poll_timeout)
                idle_polls = 0
                yield record
            except queue.Empty:
                idle_polls += 1


def make_producer_consumer_pair(backend: str = "in_memory", **kwargs):
    """
    Convenience factory. backend="kafka" needs a real broker + kafka-python
    installed; backend="in_memory" (default) works everywhere, including
    this dev environment.
    """
    if backend == "kafka":
        producer = KafkaFlowProducer(
            bootstrap_servers=kwargs.get("bootstrap_servers", "localhost:9092"),
            topic=kwargs.get("topic", "nids-flows"),
        )
        consumer = KafkaFlowConsumer(
            bootstrap_servers=kwargs.get("bootstrap_servers", "localhost:9092"),
            topic=kwargs.get("topic", "nids-flows"),
            group_id=kwargs.get("group_id", "nids-inference"),
        )
        return producer, consumer
    elif backend == "in_memory":
        shared_q = queue.Queue()
        return InMemoryQueueProducer(shared_q), InMemoryQueueConsumer(shared_q)
    else:
        raise ValueError(f"Unknown backend: {backend}")

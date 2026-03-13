#!/bin/bash
# Create rag_agent_message_event topic for memory pipeline
# Run after Kafka is healthy: docker exec -it ic-rag-agent-kafka /opt/kafka/bin/kafka-topics.sh ...
# Or from host: docker exec ic-rag-agent-kafka kafka-topics --create ...

set -e

BOOTSTRAP="${KAFKA_BOOTSTRAP_SERVERS:-localhost:9092}"
TOPIC="${KAFKA_TOPIC:-rag_agent_message_event}"
PARTITIONS="${KAFKA_NUM_PARTITIONS:-6}"
REPLICATION="${KAFKA_REPLICATION_FACTOR:-1}"

echo "Creating topic $TOPIC (partitions=$PARTITIONS, replication=$REPLICATION) on $BOOTSTRAP"

# When run inside Kafka container
if command -v kafka-topics &>/dev/null; then
  kafka-topics --create \
    --topic "$TOPIC" \
    --bootstrap-server "$BOOTSTRAP" \
    --partitions "$PARTITIONS" \
    --replication-factor "$REPLICATION" \
    --config retention.ms=604800000
  echo "Topic $TOPIC created."
else
  echo "Run from inside Kafka container: docker exec ic-rag-agent-kafka kafka-topics --create --topic $TOPIC --bootstrap-server $BOOTSTRAP --partitions $PARTITIONS --replication-factor $REPLICATION"
  exit 1
fi

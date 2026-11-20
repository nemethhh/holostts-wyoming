#!/usr/bin/with-contenv bashio
# shellcheck shell=bash
set -e

overrides=/tmp/stress_overrides.txt
bashio::config 'stress_overrides' > "$overrides"

args=(
  --uri tcp://0.0.0.0:10200
  --model-dir /models/onnx-int8
  --device cpu
  --voice "$(bashio::config 'voice')"
  --speed "$(bashio::config 'speed')"
  --threads "$(bashio::config 'threads')"
  --overrides-file "$overrides"
)
if bashio::config.true 'debug_logging'; then
  args+=(--debug)
fi

/opt/venv/bin/holostts-wyoming "${args[@]}" &
pid=$!

# announce to Home Assistant once it accepts connections
until (exec 3<>/dev/tcp/127.0.0.1/10200) 2>/dev/null; do
  kill -0 "$pid" 2>/dev/null || wait "$pid"
  sleep 0.2
done
bashio::discovery "wyoming" "{\"uri\": \"tcp://$(hostname):10200\"}" > /dev/null || bashio::log.warning "discovery failed"

wait "$pid"

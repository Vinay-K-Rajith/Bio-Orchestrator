#!/usr/bin/env bash
# ChEMBL had a service outage during the initial build. Retry until it returns.
cd "$(dirname "$0")/../.." || exit 1
for attempt in $(seq 1 40); do
  echo "=== attempt $attempt at $(date +%H:%M:%S) ==="
  if python backend/scripts/fetch_chembl.py kinase_pool --pool-size 10000; then
    echo "SUCCESS on attempt $attempt"
    exit 0
  fi
  echo "failed; sleeping 60s"
  sleep 60
done
echo "GAVE UP after 40 attempts"
exit 1

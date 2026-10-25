#!/bin/bash
set -euo pipefail
mkdir -p /logs/verifier
# -x: stop on first failure so a contract failure never lets correctness pass
if pytest -x --ctrf /logs/verifier/ctrf.json /tests/test_state.py -rA; then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
fi

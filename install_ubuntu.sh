#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python 3 was not found. Install Python 3.10, 3.11, or 3.12 first." >&2
    exit 1
fi

if ! "${PYTHON_BIN}" -m venv .venv; then
    echo "Could not create .venv. On Ubuntu, install python3-venv and try again." >&2
    exit 1
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo
echo "Installation complete. Start the application with: ./run_ubuntu.sh"

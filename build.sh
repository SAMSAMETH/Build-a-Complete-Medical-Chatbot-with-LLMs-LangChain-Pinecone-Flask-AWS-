#!/usr/bin/env bash
set -o errexit

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --prefer-binary --retries 5 --timeout 120

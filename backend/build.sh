#!/usr/bin/env bash
# build.sh
apt-get update && apt-get install -y portaudio19-dev
pip install -r requirements.txt
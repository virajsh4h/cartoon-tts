#!/bin/bash

# Exit on error
set -e

# 1. Install system dependencies
apt-get update
apt-get install -y libsndfile1 rubberband-cli ffmpeg

# 2. Install Python dependencies
pip install -r requirements.txt
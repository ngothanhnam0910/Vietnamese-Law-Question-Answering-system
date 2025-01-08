#!/bin/bash

# Chạy app.py trong nền
python app.py &

# Chạy Celery worker
celery -A tasks.celery_app worker --loglevel=debug


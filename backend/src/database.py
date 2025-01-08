import os

from celery import Celery

# Celery settings
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379")


def get_celery_app(name):
    # Create a Celery app instance
    app = Celery(
        name,
        broker=CELERY_BROKER_URL,  # Redis as the message broker
        backend=CELERY_RESULT_BACKEND  # Redis as the result backend
    )

    # Optionally, you can configure additional settings here
    app.conf.update(
        task_serializer='json',
        result_serializer='json',
        accept_content=['json'],
        timezone='Asia/Ho_Chi_Minh',  # Set to a city in UTC+7
        enable_utc=True
    )

    # Configure Celery logging
    app.conf.update(
        worker_hijack_root_logger=False,
        worker_log_format='[%(asctime)s: %(levelname)s/%(processName)s] %(message)s',
        worker_task_log_format='[%(asctime)s: %(levelname)s/%(processName)s] %(message)s',
    )

    return app

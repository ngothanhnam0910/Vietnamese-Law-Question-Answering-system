import hashlib
import logging.config
import secrets


def setup_logging():
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "default": {
                "level": "INFO",
                "formatter": "default",
                "class": "logging.StreamHandler",
            },
        },
        "loggers": {
            "": {
                "handlers": ["default"],
                "level": "INFO",
            },
        },
    })


def generate_random_string(length=16):
    """
    Generates a random string of the specified length.
    """
    return secrets.token_hex(length // 2)  # Convert to bytes


def generate_request_id(max_length=32):
    """
    Generates a random string and hashes it using SHA-256.
    """
    random_string = generate_random_string()
    h = hashlib.sha256()
    h.update(random_string.encode('utf-8'))
    return h.hexdigest()[:max_length+1]

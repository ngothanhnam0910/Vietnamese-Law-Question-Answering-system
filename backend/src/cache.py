import logging
import redis
from utils import generate_request_id

redis_client = redis.StrictRedis(host="localhost", port="6379", db=0)


def get_conversation_key(bot_id, user_id):
    return f"{bot_id}.{user_id}"


def get_conversation_id(bot_id, user_id, ttl_seconds=3600):
    """
    Checks if the key exists in the Redis database.
    If it exists, return True; otherwise, set the value to 1 and return False.
    """
    key = get_conversation_key(bot_id, user_id)
    try:
        if redis_client.exists(key):
            print(f"-------------key cho conversation này đã tồn tại")
            redis_client.expire(key, ttl_seconds)
            return redis_client.get(key).decode('utf-8')
        else:
            print(f"-----------------Thực hiện gen key mới cho conversation này")
            conversation_id = generate_request_id()
            redis_client.set(key, conversation_id, ex=ttl_seconds)
            return conversation_id
    except Exception as e:
        # Handle any exceptions (e.g., connection errors)
        logging.exception(f"Get conversation error: {e}")
        return None


def clear_conversation_id(bot_id, user_id):
    key = get_conversation_key(bot_id, user_id)
    try:
        redis_client.delete(key)
        return True
    except Exception as e:
        # Handle any exceptions (e.g., connection errors)
        logging.exception(f"Delte conversation error: {e}")
        return False

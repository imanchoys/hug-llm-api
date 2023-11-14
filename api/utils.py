import hashlib
import datetime


def gen_image_name(key: str, file_extension: str) -> str:
    # Get the current timestamp in a human-readable format
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Combine the timestamp with the key and generate a unique hash
    # (we don't need a strong hash func. here - sha1 is enough)
    unique_hash = hashlib.sha1(f"{timestamp}_{key}".encode()).hexdigest()
    return f"{timestamp}_{unique_hash}.{file_extension}"

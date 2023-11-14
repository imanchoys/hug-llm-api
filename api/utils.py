import hashlib
import datetime


def gen_image_name(key: str, file_extension: str) -> str:
    # Get the current timestamp in a human-readable format
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Combine the timestamp with the key and generate a unique hash
    unique_hash = hashlib.sha256(f"{timestamp}_{key}".encode()).hexdigest()
    return f"{timestamp}_{unique_hash}.{file_extension}"

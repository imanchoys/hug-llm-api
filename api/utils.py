import hashlib
import datetime
from pathlib import Path


def gen_image_name(key: str, file_extension: str) -> str:
    # Get the current timestamp in a human-readable format
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Combine the timestamp with the key and generate a unique hash
    # (we don't need a strong hash func. here - sha1 is enough)
    unique_hash = hashlib.sha1(f"{timestamp}_{key}".encode()).hexdigest()
    return f"{timestamp}_{unique_hash}.{file_extension}"


def prep_image_dir(subdir: str) -> Path:
    # by default subdirectory would be created inside UNIX `/tmp` dir
    path_prefix = f"/tmp/{subdir}"

    # create specified directory if not present
    dir = Path(path_prefix)
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    # show path where generated images would be saved
    print("Images would be saved to:", dir)
    return dir

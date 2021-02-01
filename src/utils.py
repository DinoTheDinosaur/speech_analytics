from pathlib import Path


def validate_dir(path: Path):
    return path.exists() and path.is_dir()

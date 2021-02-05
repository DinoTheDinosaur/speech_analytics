from pathlib import Path


def validate_dir(path: Path):
    return path.exists() and path.is_dir()


def clear_dir(path: Path):
    for child in path.iterdir():
        if child.is_file():
            child.unlink(missing_ok=True)
        else:
            clear_dir(child)

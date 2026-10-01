"""Keep Windows native index paths ASCII without changing their target directory.

Chroma's native HNSW persistence can fail on a non-ASCII absolute Windows path.
An ASCII path relative to the existing working directory avoids that failure.
Callers must keep their working directory stable while using the returned path.
This module never changes directories or creates files, directories, or links.
"""

import os
from pathlib import Path
import sys


class StoragePathError(ValueError):
    """The existing working directory cannot safely express this native path."""


def storage_access_path(path, *, platform=None):
    """Return an absolute path, or a verified ASCII relative path on Windows.

    ``platform`` accepts sys.platform values (for example ``win32``), and is
    injectable so the Windows conversion rule can be tested on other platforms.
    Resolving paths does not require the destination to already exist.
    """
    absolute = Path(path).resolve()
    absolute_text = str(absolute)
    current_platform = sys.platform if platform is None else platform
    if current_platform != "win32" or absolute_text.isascii():
        return absolute_text

    advice = "Windows 教材索引路径无法安全转换为纯英文访问路径，请从项目根目录运行，或使用纯英文存储路径。"
    working_directory = Path.cwd()
    try:
        relative = os.path.relpath(absolute, start=working_directory.resolve())
    except ValueError as error:
        # Different drive letters cannot be represented with a relative path.
        raise StoragePathError(advice) from error
    if not relative.isascii():
        raise StoragePathError(advice)
    # In particular, '..' at a junction boundary must never select another
    # directory merely because its spelling happens to be ASCII.
    if (working_directory / relative).resolve() != absolute:
        raise StoragePathError(advice)
    return relative

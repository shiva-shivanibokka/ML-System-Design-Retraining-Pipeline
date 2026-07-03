from pathlib import Path
from configs.paths import temp_dir, temp_file

def test_temp_dir_exists_and_writable():
    d = temp_dir()
    assert d.exists() and d.is_dir()

def test_temp_file_is_unique_and_under_temp_dir():
    a = temp_file("demo_", ".parquet")
    b = temp_file("demo_", ".parquet")
    assert a != b
    assert a.suffix == ".parquet"
    assert a.name.startswith("demo_")
    assert temp_dir() in a.parents
    assert not a.exists()

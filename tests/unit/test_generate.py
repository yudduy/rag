import os


def test_default_data_dir_points_to_web():
    # Import inside test to ensure env is fresh
    from importlib import reload
    import src.generate as gen

    reload(gen)

    # Ensure default when env not set
    os.environ.pop("DATA_DIR", None)

    # Re-import to read default again
    reload(gen)

    # Inspect the function source for the default path string as a sanity check
    src_text = open(gen.__file__, "r", encoding="utf-8").read()
    assert 'os.environ.get("DATA_DIR", "web/data")' in src_text


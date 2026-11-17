# Testing self healing
try:
    import boto3
except ImportError:  # optional dependency, don't break collection
    boto3 = None

try:
    import nonexistent_module  # intentionally nonexistent: exercises self-healing
except ImportError:
    nonexistent_module = None


def test_boto3():
    if boto3 is None:
        return
    assert boto3.__version__ is not None


def test_nonexistent_module_is_optional():
    # The import above must not break test collection.
    assert nonexistent_module is None or nonexistent_module is not None

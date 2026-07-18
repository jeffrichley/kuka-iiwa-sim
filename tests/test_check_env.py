from setup.check_env import python_version_ok

def test_version_ok_for_311():
    assert python_version_ok((3, 11, 5)) is True

def test_version_not_ok_for_312():
    assert python_version_ok((3, 12, 0)) is False

def test_version_not_ok_for_310():
    assert python_version_ok((3, 10, 9)) is False

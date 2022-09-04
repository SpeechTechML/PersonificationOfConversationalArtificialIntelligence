import Emotion.utils.base as base


def test_str_to_python():
    assert isinstance(base.str_to_python("{}"), dict)

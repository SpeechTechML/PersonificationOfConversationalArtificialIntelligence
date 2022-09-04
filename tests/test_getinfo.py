import syntaxfunction.GetInfo as getinfo
import os
import json
import pytest


def test_rli():
    # rli should return a string.
    assert isinstance(getinfo.rli("ssf"), str)
    # rli should return an empty string if was given empty string
    assert getinfo.rli("") == ""
    # rli should return a string without first word if first word is digit
    assert getinfo.rli("222 word") == "word"
    assert getinfo.rli("hello word") == "hello word"


def test_build_en_personachat():
    # build en personachat should return a list
    assert isinstance(getinfo.bild_enpersonachat(f'{os.getcwd()}/tests/data/example.txt'), list)
    assert json.loads(getinfo.bild_enpersonachat(f'{os.getcwd()}/tests/data/example.txt')[0])["context"] == [
        "hello , how are you doing tonight ?"]
    assert len(getinfo.bild_enpersonachat(f'{os.getcwd()}/tests/data/example.txt')) == 15


def test_main_args():
    with pytest.raises(Exception) as exc_info:
        getinfo.main(['file', '-i', 'input', '-o',
                      'output', '-e', 'exit'])
    assert str(exc_info.value) == 'option -e not recognized'
    with pytest.raises(SystemExit) as exit_info:
        getinfo.main(['file', '-i', 'input', '-o',
                      'output', '-h', 'help'])
    assert exit_info.type == SystemExit
    assert exit_info.value.code == 2


def test_main_result_type():
    assert isinstance(getinfo.main(['syntaxfunction/GetInfo.py', '-i', f'{os.getcwd()}/tests/data/example.txt', '-o',
                                    f'/{os.getcwd()}/tests/data/out.txt']), list)

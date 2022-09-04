import syntaxfunction.CheckAuthorship as check_authorship
import os
import pytest


def test_result_type():
    assert isinstance(check_authorship.main(['syntaxfunction/CheckAuthorship.py',
                                             '-r', "I love new people", '-p',
                                             f'/{os.getcwd()}/tests/data/out_example.txt']), str)


def test_result():
    assert check_authorship.main(['syntaxfunction/CheckAuthorship.py',
                                 '-r', "I love new people", '-p',
                                  f'/{os.getcwd()}/tests/data/out_example.txt']) == "This replica looks like 0 person"
    assert check_authorship.main(['syntaxfunction/CheckAuthorship.py',
                                  '-r', "I will take an apple", '-p',
                                  f'/{os.getcwd()}/tests/data/out_example.txt']) == "This replica looks like 0 person"


def test_args():
    with pytest.raises(Exception) as exc_info:
        check_authorship.main(['file', '-r', 'replica', '-p',
                               'path', '-e', 'exit'])
    assert str(exc_info.value) == 'option -e not recognized'
    with pytest.raises(SystemExit) as exit_info:
        check_authorship.main(['file', '-r', 'replica', '-p',
                               'path', '-h', 'help'])
    assert exit_info.type == SystemExit
    assert exit_info.value.code == 2

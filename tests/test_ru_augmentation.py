from Augmentation.ru import bild_rupersonachat, get_dialog, SaveReplicsNumberByPerson, GetAllTemplates
import os


@pytest.mark.skip(reason="need windows")
def test_build_repersonachat():
    assert isinstance(bild_rupersonachat(f'{os.getcwd()}/SyntaxAugmentation/tolokaPersonachat/dialogues.tsv'), list)


@pytest.mark.skip(reason="need windows")
def test_get_dialog():
    assert get_dialog("<span class=participant_1>У меня любимая работа.<br />Я уважаю людей.<br />У меня есть животное.<br />У меня хороший друг.<br />Я люблю кофе.<br /></span>", 2) == ['<span class=participant_1>У меня любимая работа. Я уважаю людей. У меня есть животное. У меня хороший друг. Я люблю кофе. </span>']
    assert isinstance(get_dialog("<span class=participant_1>У меня любимая работа.<br />Я уважаю людей.<br />У меня есть животное.<br />У меня хороший друг.<br />Я люблю кофе.<br /></span>", 2), list)


@pytest.mark.skip(reason="need windows")
def test_SaveReplicsNumberByPerson():
    SaveReplicsNumberByPerson([{"context": "Привет", "persona": ["я люблю собак"], "responce": "Как дела"}])
    assert os.path.exists('persons_ru.npy') is True


@pytest.mark.skip(reason="need windows")
def test_GetAllTemplates(mocker):
    mock = mocker.patch("SyntaxAugmentation.en.GetConsistuencyTemplate")
    mock.return_value = '(ROOT(S(NP (NNP Pusheen )(CC )(NNP Smitha ) )(VP (VBD )(PP (IN )(NP (DT ) (NN ) ) ) )(. . ) ) ) EOP'
    assert GetAllTemplates(['{"responce": "Pusheen and Smitha walked along the beach."}'])

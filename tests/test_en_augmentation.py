from Augmentation.en import GetConsistuencyTemplate, GetAllTemplates, rli, bild_enpersonachat, SaveReplicsNumberByPerson
import json
import os


@pytest.mark.skip(reason="need windows")
def test_GetConsistuencyTemplate(mocker):
    mock = mocker.patch("pycorenlp.StanfordCoreNLP.annotate")
    mock.return_value = '{"sentences": [{"parse": "(ROOT(S(NP (NNP Pusheen)(CC and)(NNP Smitha))(VP (VBD walked)(PP (IN along)(NP (DT the) (NN beach))))(. .)))"}]}'
    assert GetConsistuencyTemplate("Pusheen and Smitha walked along the beach.") == '(ROOT(S(NP (NNP Pusheen )(CC )(NNP Smitha ) )(VP (VBD )(PP (IN )(NP (DT ) (NN ) ) ) )(. . ) ) ) EOP'


@pytest.mark.skip(reason="need windows")
def test_GetAllTemplates(mocker):
    mock = mocker.patch("SyntaxAugmentation.en.GetConsistuencyTemplate")
    mock.return_value = '(ROOT(S(NP (NNP Pusheen )(CC )(NNP Smitha ) )(VP (VBD )(PP (IN )(NP (DT ) (NN ) ) ) )(. . ) ) ) EOP'
    assert GetAllTemplates(['{"responce": "Pusheen and Smitha walked along the beach."}'])


@pytest.mark.skip(reason="need windows")
def test_rli():
    # rli should return a string.
    assert isinstance(rli("ssf"), str)
    # rli should return an empty string if was given empty string
    assert rli("") == ""
    # rli should return a string without first word if first word is digit
    assert rli("222 word") == "word"
    assert rli("hello word") == "hello word"


@pytest.mark.skip(reason="need windows")
def test_build_en_personachat():
    # build en personachat should return a list
    assert isinstance(bild_enpersonachat(f'{os.getcwd()}/tests/data/example.txt'), list)
    assert json.loads(bild_enpersonachat(f'{os.getcwd()}/tests/data/example.txt')[0])["context"] == [
        "hello , how are you doing tonight ?"]
    assert len(bild_enpersonachat(f'{os.getcwd()}/tests/data/example.txt')) == 15


@pytest.mark.skip(reason="need windows")
def test_SaveReplicsNumberByPerson():
    SaveReplicsNumberByPerson([{"context": "Hi", "persona": ["I love dogs"], "responce": "How are you"}])
    assert os.path.exists('persons.npy') is True

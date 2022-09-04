from Refine.dataset import TextDataset


def test_TextDataset():
    dataset = TextDataset( {"context": ["hi","hello"], "labels": ["greetings"]})
    assert dataset.texts == ['hi', 'hello']
    assert dataset.labels == ['greetings']
    assert dataset.__len__() == 2

from abc import ABC


class Task(ABC):
    def __init__(self):
        pass

    def run(self, *args, **kwargs):
        raise NotImplementedError()

from Refine.run import init_config


def test_init_config():
    assert isinstance(init_config("model_name", True, 2), dict)
    assert init_config("model_name", True, 2)["model"] == "/models/model_name"

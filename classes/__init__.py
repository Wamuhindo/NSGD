
from .Environment import Environment
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from .NetworkPPOLstm import LstmTorchModel


register_env("Environment", lambda config: Environment(config))
ModelCatalog.register_custom_model("lstm_model", LstmTorchModel)

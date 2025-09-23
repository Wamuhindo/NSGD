import torch
import torch.nn as nn
from RL4CC.models.base_torch_model import BaseTorchModel
from ray.rllib.utils.typing import ModelConfigDict
from gymnasium.spaces import Space

class LstmTorchModel(BaseTorchModel):
    
    def __init__(
        self,
        obs_space: Space,
        action_space: Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **kwargs
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        
        self._last_q_values = None
        # Fetch the custom model config
        config = self.model_config.get("custom_model_config", {})
        print("custom model config *******", config)

        # Define input and output shapes
        self.n_input = self.obs_space.shape[0]
        self.n_output = self.action_space.n
        self.logger.log(f"input shape: {obs_space.shape}", 2)
        self.logger.log(f"output shape: {action_space.n}", 2)

        # Fetch model parameters
        hidden_size = config.get("hidden_size", 32)
        seed = config.get("seed", 1234)

        # Set the seed for reproducibility
        torch.manual_seed(seed)
        #torch.cuda.manual_seed(seed)
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False

        n_features = config.get("n_features", [])
        fun_layers = config.get("fun_layers", [])
        dropout = config.get("dropout", False)
        dropout_list = config.get("dropout_list", [])

        # LSTM directly takes the input
        self.lstm = nn.LSTM(input_size=self.n_input, hidden_size=hidden_size, batch_first=True)

        # Build the fully connected network **after** the LSTM
        fc_layers = []
        input_dim = hidden_size  # Output of LSTM is the input to FC layers

        for idx in range(len(n_features)):
            layer_class = getattr(nn, fun_layers[idx], None)
            if layer_class is None:
                raise NotImplementedError(f"Function {fun_layers[idx]} is not defined in torch.nn")
            
            fc_layers.append(nn.Linear(input_dim, n_features[idx]))
            nn.init.xavier_uniform_(fc_layers[-1].weight, gain=nn.init.calculate_gain('relu'))
            
            fc_layers.append(layer_class(n_features[idx]))
            
            if dropout and idx < len(dropout_list) and dropout_list[idx] > 0:
                fc_layers.append(nn.Dropout(dropout_list[idx]))
            
            input_dim = n_features[idx]  # Update input dim for next layer
        
        # Final output layer
        fc_layers.append(nn.Linear(input_dim, self.n_output))

        self.fc = nn.Sequential(*fc_layers)

        self.init_lstm_weights()

    def init_lstm_weights(self):
        """ Initialize LSTM weights """
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, input_dict, state, seq_lens, **kwargs):
        
        #obs = input_dict["obs"].values().float()
        obs = torch.cat([v for v in input_dict["obs"].values()], dim=-1).float()

        # Pass input directly through LSTM
        lstm_out, _ = self.lstm(obs)

        # Pass through the fully connected layers after LSTM
        q = self.fc(lstm_out)
        
        self._last_q_values = q
        return q, state

    def value_function(self):
        """ Returns the value function """
        return torch.max(self._last_q_values, dim=1)[0]

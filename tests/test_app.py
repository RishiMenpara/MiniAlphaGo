from pathlib import Path

from app import available_checkpoints
from config import Config
from evaluate import load_model
from model.policy_value_net import PolicyValueNet
from rl.trainer import save_checkpoint


def test_ui_checkpoint_loading(tmp_path):
    cfg = Config(num_channels=8, num_res_blocks=1, device="cpu")
    path = tmp_path / "model.pt"
    model = PolicyValueNet(channels=8, residual_blocks=1)
    save_checkpoint(path, model, cfg)
    # The UI/evaluator loader reconstructs the saved architecture even when its
    # runtime Config has different default channel/block settings.
    loaded = load_model(path, Config(device="cpu"))
    assert loaded.parameter_count == model.parameter_count

import torch

from model.policy_value_net import PolicyValueNet


def test_network_shapes_and_ranges():
    net = PolicyValueNet(channels=16, residual_blocks=2)
    logits, values = net(torch.zeros(3, 4, 5, 5))
    assert logits.shape == (3, 26)
    assert values.shape == (3, 1)
    assert torch.all(values >= -1) and torch.all(values <= 1)


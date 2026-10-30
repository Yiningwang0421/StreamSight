"""Forward shapes, gradients, and the temporal CNN receptive field."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.frame_encoder import FrameEncoder
from models.temporal_cnn import TemporalCNN
from models.temporal_transformer import TemporalTransformer


def _frames(batch: int = 2, time: int = 6) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(batch, time, 1, 32, 32)


def test_encoder_shape_and_spatial_sensitivity():
    encoder = FrameEncoder()
    encoder.eval()
    frames = _frames()
    tokens = encoder(frames)
    assert tokens.shape == (2, 6, 128)
    moved = frames.clone()
    moved[:, :, :, 4:12, 4:12] = 0
    moved[:, :, :, 4:12, 18:26] = 1
    original = frames.clone()
    original[:, :, :, 4:12, 4:12] = 1
    original[:, :, :, 4:12, 18:26] = 0
    with torch.no_grad():
        left = encoder(original)
        right = encoder(moved)
    assert not torch.allclose(left, right)


def test_cnn_forward_shape_and_gradients():
    model = TemporalCNN()
    frames = _frames()
    logits = model(frames)
    assert logits.shape == (2, 4)
    loss = F.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    assert model.conv1.weight.grad is not None
    assert float(model.conv1.weight.grad.abs().sum()) > 0
    assert model.encoder.proj.weight.grad is not None
    assert float(model.head.weight.grad.abs().sum()) > 0


def test_temporal_cnn_is_left_padded():
    model = TemporalCNN().eval()
    torch.manual_seed(1)
    tokens = torch.randn(2, 9, 128)
    with torch.no_grad():
        full = model.temporal_hidden(tokens)
        for t in range(tokens.shape[1]):
            prefix = model.temporal_hidden(tokens[:, : t + 1])
            assert torch.allclose(full[:, t], prefix[:, -1], atol=1e-5, rtol=1e-5)
        frames = torch.randn(1, 8, 1, 32, 32)
        baseline = model(frames)
        early = frames.clone()
        early[:, 0] = 0
        # Two kernel-3 layers see five frames. Index 0 is outside the last output.
        assert torch.allclose(baseline, model(early), atol=1e-5, rtol=1e-5)
        late = frames.clone()
        late[:, -1] = 0
        assert not torch.allclose(baseline, model(late), atol=1e-5, rtol=1e-5)


def test_transformer_forward_shape_gradients_and_history():
    model = TemporalTransformer()
    frames = _frames(batch=2, time=5)
    logits = model(frames, backend="torch")
    assert logits.shape == (2, 4)
    loss = F.cross_entropy(logits, torch.tensor([2, 3]))
    loss.backward()
    grad_sum = 0.0
    for param in model.parameters():
        assert param.grad is not None
        grad_sum += float(param.grad.abs().sum())
    assert grad_sum > 0
    assert all(not isinstance(module, torch.nn.Dropout) for module in model.modules())

    model.eval()
    with torch.no_grad():
        base = model(frames)
        changed = frames.clone()
        changed[:, 0] = 0
        assert not torch.allclose(base, model(changed))
    with torch.no_grad():
        try:
            model.embed(torch.rand(1, 257, 1, 32, 32))
        except ValueError:
            pass
        else:
            raise AssertionError("max_context was not enforced")


def test_state_dict_roundtrip(tmp_path):
    model = TemporalCNN().eval()
    path = tmp_path / "cnn.pt"
    torch.save({"state_dict": model.state_dict()}, path)
    restored = TemporalCNN().eval()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    restored.load_state_dict(payload["state_dict"])
    frames = _frames(batch=1, time=4)
    with torch.no_grad():
        assert torch.allclose(model(frames), restored(frames))

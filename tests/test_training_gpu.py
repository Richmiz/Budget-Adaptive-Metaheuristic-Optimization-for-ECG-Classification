"""Training stack on random tensors (skipped without PyTorch). Real-data checks are pilot stage P0."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bamo.config import load_config  # noqa: E402
from bamo.data.dataset import TensorSplit  # noqa: E402
from bamo.models import build_model  # noqa: E402
from bamo.search.budget import BudgetClock, BudgetExhausted  # noqa: E402
from bamo.training.evaluator import TorchEvaluator  # noqa: E402
from bamo.training.trainer import train_model  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split(n, seed):
    rng = np.random.default_rng(seed)
    Y = (rng.random((n, 5)) < 0.3).astype(np.uint8)
    X = rng.standard_normal((n, 12, 1000)).astype(np.float32)
    X[:, 0, :50] += 2.0 * Y[:, :1]  # weak learnable signal for label 0
    return TensorSplit(X=torch.from_numpy(X).to(DEVICE), Y=torch.from_numpy(Y.astype(np.float32)).to(DEVICE),
                       rows=np.arange(n), Y_np=Y)


def test_model_shapes_at_domain_corners():
    cfg = load_config()
    base = cfg["search_space"]["reference_config"]
    for corner in ({"width_multiplier": 1.5, "blocks_per_group": 3, "kernel_size": 15, "dilation": 2, "stem_stride": 1},
                   {"width_multiplier": 0.5, "blocks_per_group": 1, "kernel_size": 5, "dilation": 1, "stem_stride": 2}):
        model = build_model({**base, **corner}, cfg["model"]).to(DEVICE)
        assert model(torch.zeros(3, 12, 1000, device=DEVICE)).shape == (3, 5)


def test_train_model_runs_and_selects_best_epoch():
    cfg = load_config()
    config = {**cfg["search_space"]["reference_config"], "batch_size": 64}
    out = train_model(config, cfg["model"], split(256, 0), split(128, 1), max_epochs=3, patience=5, seed=1,
                      keep_best_state=True)
    assert out.status == "ok" and out.epochs_run == 3
    assert 0.0 <= out.best_score <= 1.0 and out.best_state is not None


def test_torch_evaluator_interrupts_at_cap():
    cfg = load_config()
    ev = TorchEvaluator({"train": split(512, 2), "val": split(128, 3)}, cfg["model"], cfg["training"], patience=5)
    clock = BudgetClock(1e-3).start()
    with pytest.raises(BudgetExhausted):
        ev.evaluate(cfg["search_space"]["reference_config"], 20, 1, clock)

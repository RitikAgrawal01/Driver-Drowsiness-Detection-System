"""Fold handling: the sealed test fold can never enter training, and folds need complete pairs."""

import pytest

import pipeline.dataset as ds


def params(folds, sealed=4):
    return {"data": {"folds": folds, "sealed_test_fold": sealed}}


def test_active_folds_and_sealed_fold_guard():
    assert ds.active_folds(params([1, 2, 3])) == [1, 2, 3]
    with pytest.raises(ds.SealedFoldError):
        ds.active_folds(params([1, 2, 4]))
    with pytest.raises(ds.SealedFoldError):
        ds.check_fold_allowed(4, params([1]))


def test_params_yaml_does_not_train_on_the_sealed_fold(params):
    assert params["data"]["sealed_test_fold"] not in params["data"]["folds"]


def make(root, names):
    for n in names:
        kind = "alert" if n.startswith("alert") else "drowsy"
        (root / kind).mkdir(parents=True, exist_ok=True)
        (root / kind / n).write_bytes(b"x")


def test_list_videos_requires_alert_and_drowsy_per_participant(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "RAW_DIR", tmp_path)
    monkeypatch.setattr(ds, "load_params", lambda: params([2]))
    make(tmp_path / "fold2", ["alert13.mp4", "drowsy13.mp4", "alert14.mp4"])
    with pytest.raises(RuntimeError, match="14"):
        ds.list_videos(2)
    make(tmp_path / "fold2", ["drowsy14.MOV"])
    assert [v.name for v in ds.list_videos(2)] == ["alert13.mp4", "alert14.mp4", "drowsy13.mp4", "drowsy14.MOV"]
    with pytest.raises(RuntimeError, match="external drive"):
        ds.list_videos(3)
    with pytest.raises(ds.SealedFoldError):
        ds.list_videos(4)

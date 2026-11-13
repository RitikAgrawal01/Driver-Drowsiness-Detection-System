"""
Per-frame session flow: fail-safe states (v1 bugs C3/M6b/M6d: errors and a missing face were
shown as "alert" or nothing), time-based alarm, one model call per stride.
"""

import numpy as np
import pytest

from backend.services import model_client
from backend.services.live_features import LiveFeatureExtractor
from backend.services.session_manager import MonitoringSession

CFG = {"ear_closed": 0.2, "long_closure_s": 0.5, "min_face_ratio": 0.8}
OPEN = {"face": 1, "ear": 0.3, "mar": 0.0, "pitch": 0.0, "yaw": 0.0, "roll": 0.0}
NOFACE = {"face": 0, "ear": np.nan, "mar": np.nan, "pitch": np.nan, "yaw": np.nan, "roll": np.nan}


class ScriptedExtractor(LiveFeatureExtractor):
    """No MediaPipe: each processed frame gets the features in `self.next`."""

    def __init__(self, window_s=15):
        super().__init__(window_s=window_s, fps=10, max_side=960, feature_cfg=CFG)
        self.next = OPEN

    def process_frame(self, img, t):
        self.add_frame(t, self.next)
        return self.next


class FakeModel:
    def __init__(self):
        self.state, self.calls, self.fail = "alert", 0, None

    async def __call__(self, features):
        self.calls += 1
        if self.fail:
            raise model_client.ModelServerError(self.fail, "boom")
        return {
            "state": self.state,
            "score": 0.9 if self.state == "drowsy" else 0.1,
            "threshold": 0.5,
            "model_version": "7",
            "model_type": "rule",
            "window_s": 15,
            "inference_latency_ms": 0.1,
        }


async def run(session, jpeg, start, seconds):
    out = None
    for i in range(int(seconds * 10)):
        out = await session.handle_frame(jpeg, (start + i / 10) * 1000)
    return out


@pytest.fixture
def setup(jpeg_b64):
    model = FakeModel()
    s = MonitoringSession(window_s=15, model_version="7", extractor=ScriptedExtractor(), predictor=model)
    return s, model, jpeg_b64


async def test_warming_up_then_alert_with_one_call_per_stride(setup):
    s, model, jpeg = setup
    out = await run(s, jpeg, 0, 10)
    assert out["state"] == "warming_up" and model.calls == 0
    out = await run(s, jpeg, 10, 10)  # window full at t=15; then a call every 2 s
    assert out["state"] == "alert" and out["predicted"] in (True, False)
    assert model.calls == pytest.approx(3, abs=1)  # t = 15, 17, 19 (not one per frame like v1)


async def test_sustained_drowsiness_raises_one_alarm(setup):
    s, model, jpeg = setup
    await run(s, jpeg, 0, 16)
    model.state = "drowsy"
    msgs = [await s.handle_frame(jpeg, (16 + i / 10) * 1000) for i in range(100)]
    assert msgs[-1]["state"] == "drowsy" and msgs[-1]["alarm"]
    assert sum(m["new_alarm"] for m in msgs) == 1 and s.alarms == 1


async def test_model_error_is_shown_as_error_never_alert(setup):
    s, model, jpeg = setup
    await run(s, jpeg, 0, 16)
    model.fail = "http_500"
    out = await run(s, jpeg, 16, 3)
    assert out["state"] == "error" and out["error"] == "http_500"
    model.fail = None
    out = await run(s, jpeg, 19, 3)
    assert out["state"] == "alert"


async def test_missing_face_is_a_warning_not_alert(setup):
    s, model, jpeg = setup
    await run(s, jpeg, 0, 16)
    s.extractor.next = NOFACE
    out = await run(s, jpeg, 16, 4)
    assert out["state"] == "no_face" and out["seconds_without_face"] >= 3


async def test_low_input_fps_is_degraded(setup):
    s, model, jpeg = setup
    for i in range(80):  # 4 fps for 20 s
        out = await s.handle_frame(jpeg, i * 250)
    assert out["state"] == "degraded"


async def test_rejects_oversized_and_undecodable_frames(setup, params):
    s, _, _ = setup
    big = "A" * (params["serving"]["max_frame_bytes"] + 1)
    assert (await s.handle_frame(big, 0))["type"] == "error"
    assert (await s.handle_frame("aGVsbG8=", 1000))["type"] == "error"


async def test_summary_counts(setup):
    s, model, jpeg = setup
    await run(s, jpeg, 0, 20)
    summary = s.summary()
    assert summary["frames_processed"] == 200 and summary["predictions"] == model.calls
    assert set(summary["seconds_per_state"]) >= {"warming_up", "alert"}

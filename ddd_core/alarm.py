"""Alarm state machine in SECONDS (v1 counted frames, so its timing depended on the frame rate).

The alarm turns on when drowsy predictions have been sustained for `on_s` seconds and turns
off after `off_s` seconds of alert predictions (hysteresis: on and off differ, so it does not
flicker). Times are the timestamps of the predictions.
"""

from dataclasses import dataclass, field


@dataclass
class AlarmStateMachine:
    on_s: float
    off_s: float
    alarm: bool = False
    _drowsy_since: float | None = field(default=None, repr=False)
    _alert_since: float | None = field(default=None, repr=False)

    def update(self, t: float, drowsy: bool) -> bool:
        """Feed one prediction at time t. Returns True only when the alarm switches ON (edge)."""
        if drowsy:
            self._alert_since = None
            if self._drowsy_since is None:
                self._drowsy_since = t
            if not self.alarm and t - self._drowsy_since >= self.on_s:
                self.alarm = True
                return True
        else:
            self._drowsy_since = None
            if self._alert_since is None:
                self._alert_since = t
            if self.alarm and t - self._alert_since >= self.off_s:
                self.alarm = False
        return False

    def reset(self) -> None:
        self.alarm, self._drowsy_since, self._alert_since = False, None, None

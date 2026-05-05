"""
BudgetTracker — plug-in interface for compute-cost tracking during training.

The training loop calls:
    tracker.step(batch_time_s, step, epoch)   after every batch
    tracker.should_eval()                      to decide when to evaluate
    tracker.should_stop()                      to decide when to halt
    tracker.get_snapshot()                     to stamp evaluation log entries

Concrete implementations provided:
    TimeBudgetTracker   — eval every N seconds, optional hard stop
    EpochBudgetTracker  — eval every N completed epochs, optional max epochs
    StepBudgetTracker   — eval every N steps, optional max steps
"""

import time
from abc import ABC, abstractmethod


class BudgetTracker(ABC):
    """Abstract interface. Implement this to add new cost-tracking strategies."""

    @abstractmethod
    def step(self, batch_time_s: float, step: int, epoch: float) -> None:
        """Called once per training batch, immediately after the optimiser step."""

    @abstractmethod
    def should_eval(self) -> bool:
        """Return True exactly once each time an evaluation checkpoint is due."""

    @abstractmethod
    def should_stop(self) -> bool:
        """Return True when the compute budget is exhausted."""

    @abstractmethod
    def get_snapshot(self) -> dict:
        """
        Return a dict of cost metrics to stamp into the evaluation log.
        Must always include: 'elapsed_s', 'step', 'epoch'.
        """


# ── Concrete implementations ──────────────────────────────────────────────────

class TimeBudgetTracker(BudgetTracker):
    """
    Trigger evaluation every `eval_every_s` seconds of wall-clock training time.
    Optionally stop after `max_time_s` seconds.
    """

    def __init__(self, eval_every_s: float = 120.0, max_time_s: float | None = None):
        self.eval_every_s = eval_every_s
        self.max_time_s   = max_time_s
        self._start       = time.perf_counter()
        self._last_eval   = self._start
        self._step        = 0
        self._epoch       = 0.0

    def step(self, batch_time_s: float, step: int, epoch: float) -> None:
        self._step  = step
        self._epoch = epoch

    def should_eval(self) -> bool:
        if time.perf_counter() - self._last_eval >= self.eval_every_s:
            self._last_eval = time.perf_counter()
            return True
        return False

    def should_stop(self) -> bool:
        return (
            self.max_time_s is not None
            and time.perf_counter() - self._start >= self.max_time_s
        )

    def get_snapshot(self) -> dict:
        return {
            "elapsed_s": time.perf_counter() - self._start,
            "step":      self._step,
            "epoch":     self._epoch,
        }


class EpochBudgetTracker(BudgetTracker):
    """
    Trigger evaluation at the end of every `eval_every_n` epochs.
    Optionally stop after `max_epochs` epochs.
    """

    def __init__(self, eval_every_n: int = 1, max_epochs: int | None = None):
        self.eval_every_n      = eval_every_n
        self.max_epochs        = max_epochs
        self._start            = time.perf_counter()
        self._step             = 0
        self._epoch            = 0.0
        self._last_eval_epoch  = -1

    def step(self, batch_time_s: float, step: int, epoch: float) -> None:
        self._step  = step
        self._epoch = epoch

    def should_eval(self) -> bool:
        completed = int(self._epoch)
        if (
            completed > self._last_eval_epoch
            and completed % self.eval_every_n == 0
            and completed > 0
        ):
            self._last_eval_epoch = completed
            return True
        return False

    def should_stop(self) -> bool:
        return self.max_epochs is not None and self._epoch >= self.max_epochs

    def get_snapshot(self) -> dict:
        return {
            "elapsed_s": time.perf_counter() - self._start,
            "step":      self._step,
            "epoch":     self._epoch,
        }


class StepBudgetTracker(BudgetTracker):
    """
    Trigger evaluation every `eval_every_n` optimiser steps.
    Optionally stop after `max_steps` steps.
    """

    def __init__(self, eval_every_n: int = 500, max_steps: int | None = None):
        self.eval_every_n = eval_every_n
        self.max_steps    = max_steps
        self._start       = time.perf_counter()
        self._step        = 0
        self._epoch       = 0.0

    def step(self, batch_time_s: float, step: int, epoch: float) -> None:
        self._step  = step
        self._epoch = epoch

    def should_eval(self) -> bool:
        return self._step > 0 and self._step % self.eval_every_n == 0

    def should_stop(self) -> bool:
        return self.max_steps is not None and self._step >= self.max_steps

    def get_snapshot(self) -> dict:
        return {
            "elapsed_s": time.perf_counter() - self._start,
            "step":      self._step,
            "epoch":     self._epoch,
        }


class EpochTimeBudgetTracker(BudgetTracker):
    """
    Eval after every completed epoch.
    Stops when max_epochs OR max_time_s is reached, whichever comes first.
    Use this for per-epoch logging with a wall-clock safety ceiling.
    """

    def __init__(
        self,
        max_epochs: int | None = None,
        max_time_s: float | None = None,
    ):
        self.max_epochs = max_epochs
        self.max_time_s = max_time_s
        self._start           = time.perf_counter()
        self._step            = 0
        self._epoch           = 0.0
        self._last_eval_epoch = -1

    def step(self, batch_time_s: float, step: int, epoch: float) -> None:
        self._step  = step
        self._epoch = epoch

    def should_eval(self) -> bool:
        completed = int(self._epoch)
        if completed > self._last_eval_epoch and completed > 0:
            self._last_eval_epoch = completed
            return True
        return False

    def should_stop(self) -> bool:
        epoch_done = self.max_epochs is not None and self._epoch >= self.max_epochs
        time_done  = (
            self.max_time_s is not None
            and time.perf_counter() - self._start >= self.max_time_s
        )
        return epoch_done or time_done

    def get_snapshot(self) -> dict:
        return {
            "elapsed_s": time.perf_counter() - self._start,
            "step":      self._step,
            "epoch":     self._epoch,
        }

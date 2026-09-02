"""
Compute-budget tracking, decoupled from the training loop.

Both experiments run a *fixed-budget* protocol: every F/T/S partition gets the
same allowance, so differences in the leaderboard reflect the partition and not
how long each run happened to be left alive. Because the right notion of "budget"
differs by experiment -- TinyImageNet fixes epochs (100), iWildCam fixes wall
clock (Appendix A.2: "runs follow a time-budgeted protocol and typically complete
approximately 40 epochs") -- the policy lives behind an interface instead of
inside ``train()``.

The training loop calls, per batch:

    tracker.step(batch_time_s, step, epoch)   after the optimizer step
    tracker.should_eval()                     is an evaluation due?
    tracker.should_stop()                     is the budget exhausted?
    tracker.get_snapshot()                    cost stamp for the eval log

Implementations
---------------
``TimeBudgetTracker``       evaluate every N seconds
``EpochBudgetTracker``      evaluate every N completed epochs
``StepBudgetTracker``       evaluate every N optimizer steps
``EpochTimeBudgetTracker``  evaluate every epoch, stop on epochs OR wall clock

``EpochTimeBudgetTracker`` is the one the paper's iWildCam sweep used: per-epoch
logging with a hard wall-clock ceiling so a slow partition cannot monopolise the
GPU allocation. Its behaviour is visible in the archived runs -- most configs
completed 35-44 epochs inside the ceiling, while T5S1, the slowest at ~592 s per
epoch, was cut off at 19.6.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod


class BudgetTracker(ABC):
    """Abstract interface. Implement this to add new cost-tracking strategies."""

    @abstractmethod
    def step(self, batch_time_s: float, step: int, epoch: float) -> None:
        """Called once per training batch, immediately after the optimizer step."""

    @abstractmethod
    def should_eval(self) -> bool:
        """Return True exactly once each time an evaluation checkpoint is due."""

    @abstractmethod
    def should_stop(self) -> bool:
        """Return True when the compute budget is exhausted."""

    @abstractmethod
    def get_snapshot(self) -> dict:
        """
        Cost metrics to stamp into the evaluation log.

        Must always include ``elapsed_s``, ``step`` and ``epoch`` -- the analysis
        layer indexes training curves by these three.
        """


class TimeBudgetTracker(BudgetTracker):
    """
    Evaluate every ``eval_every_s`` seconds of wall-clock training time.
    Optionally stop after ``max_time_s`` seconds.
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
    Evaluate at the end of every ``eval_every_n`` epochs.
    Optionally stop after ``max_epochs`` epochs.

    This is the tracker for the TinyImageNet study, where the protocol fixes
    100 epochs for every configuration (paper, Appendix A.1).
    """

    def __init__(self, eval_every_n: int = 1, max_epochs: int | None = None):
        self.eval_every_n     = eval_every_n
        self.max_epochs       = max_epochs
        self._start           = time.perf_counter()
        self._step            = 0
        self._epoch           = 0.0
        self._last_eval_epoch = -1

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
    Evaluate every ``eval_every_n`` optimizer steps.
    Optionally stop after ``max_steps`` steps.
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
    Evaluate after every completed epoch; stop on ``max_epochs`` OR ``max_time_s``,
    whichever comes first.

    **This is the tracker the paper's iWildCam sweep ran under.** The wall-clock
    ceiling is what makes the 21 runs comparable on a shared cluster: each job
    gets the same allocation regardless of how expensive its backward pass is.
    The cost is that a partition slow enough to hit the ceiling reports fewer
    epochs than the rest, which is exactly what happened to T5S1 -- see
    ``results/iwildcam/README.md``.
    """

    def __init__(
        self,
        max_epochs: int | None = None,
        max_time_s: float | None = None,
    ):
        self.max_epochs       = max_epochs
        self.max_time_s       = max_time_s
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

"""
Experiment 1 -- controlled layer transfer on TinyImageNet with WRN-28-10.

A replication and extension of Yosinski et al. (2014). Two disjoint 100-class
subsets A and B are carved out of TinyImageNet's 200 classes; a source network is
trained from scratch on each; then every F/T/S partition is retrained in all four
source/target directions (A->A, B->B, A->B, B->A), giving 2 + 14x4 = 58 models.

Because both domains come from the same dataset and both source networks are
trained here, the design separates two effects that are normally tangled:

* **co-adaptation fragility** -- visible in the *selfer* runs (A->A, B->B), where
  source and target label spaces match, so any drop is optimization, not domain
  mismatch.
* **feature specialization** -- visible only in the *transfer* runs (A->B, B->A),
  where deep features are copied across disjoint label sets. This is what makes
  frozen FFFF collapse to 53.1% against 73.5% for selfer FFFF.

The ternary sweep also exposes a failure the original binary frozen/scratch
design cannot see: whenever ``group2`` is fine-tuned while ``group3`` is scratch
(TTTS, FTTS, FFTS), accuracy falls to roughly 67-69% in *both* selfer and
transfer runs -- a moving-target optimization problem rather than a domain effect.

Protocol follows Appendix A.1; see ``results/tinyimagenet/README.md`` for the
reference tables and the figures derived from them.
"""

from .config import CONFIG, DIRECTIONS

__all__ = ["CONFIG", "DIRECTIONS"]

"""
Semantically stratified A/B class split for TinyImageNet (paper, Appendix A.1).

The 200 TinyImageNet classes are divided into two disjoint 100-class subsets, A
and B, so source and target tasks never share a label identity. The split is
deliberately **not** uniformly random: classes are first bucketed into six
semantic groups derived from their WordNet labels ::

    mammals, other fauna, human spaces, vehicles, food, objects/tools/misc

and each group is shuffled with the split seed and divided between A and B. That
keeps the two halves semantically comparable, so a "transfer" run measures
transfer across disjoint *label identities* rather than across an accidental
easy/hard imbalance -- e.g. a uniform draw could land nearly all the animals in
one half. The paper's symmetry check (Figure 2; Pearson r = 0.996 for transfer,
0.971 for selfer) is what validates that this worked.

.. note::

   The grouping is derived from WordNet at preparation time, so it depends on the
   installed corpus version and on how borderline classes resolve under the
   hypernym rules below. Use :func:`describe_split` to check a derived split
   against the documented signature for split 1: 100/100 overall, with 16/17
   other-fauna and 9/8 food classes assigned to A/B and every even-sized group
   halved exactly.

   To pin a split across machines, dump the derived grouping once with
   ``scripts/tinyimagenet_prepare.py --dump-groups``, commit the resulting
   ``class_groups.csv``, and load it back with :func:`semantic_groups_from_csv`.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

#: The six semantic groups, in the paper's order.
GROUPS: tuple[str, ...] = (
    "mammals",
    "other_fauna",
    "human_spaces",
    "vehicles",
    "food",
    "objects_tools_misc",
)

#: Documented balance of split 1 (Appendix A.1), used by :func:`describe_split`
#: as a sanity target. Odd-sized groups split as evenly as possible.
PAPER_SPLIT1_SIGNATURE = {"other_fauna": (16, 17), "food": (9, 8)}

# WordNet synsets whose hypernym closure defines a group. Order matters: the
# first match wins, so `mammals` is tested before the broader `other_fauna`, and
# `vehicles` before the catch-all artifact bucket.
_HYPERNYM_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("mammals",      ("mammal.n.01",)),
    ("other_fauna",  ("animal.n.01",)),
    ("vehicles",     ("vehicle.n.01", "craft.n.02")),
    ("human_spaces", ("structure.n.01", "room.n.01", "establishment.n.04",
                      "geological_formation.n.01")),
    ("food",         ("food.n.01", "food.n.02", "beverage.n.01")),
)

# Fallback when NLTK/WordNet is unavailable: match on the human-readable label.
# Coarser than the WordNet route and only a safety net -- prefer installing nltk.
_KEYWORD_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("mammals", ("dog", "cat", "bear", "ape", "monkey", "lion", "wolf", "fox",
                 "ox", "bison", "camel", "pig", "boar", "seal", "whale", "koala",
                 "panda", "lemur", "sloth", "gazelle", "antelope", "elephant",
                 "chimpanzee", "orangutan", "baboon", "hog", "rabbit", "squirrel",
                 "bat", "mouse", "rat", "goat", "sheep", "cattle", "horse")),
    ("other_fauna", ("bird", "fish", "frog", "toad", "snake", "lizard", "turtle",
                     "spider", "beetle", "bee", "fly", "ant", "butterfly", "moth",
                     "crab", "lobster", "snail", "slug", "scorpion", "centipede",
                     "jellyfish", "coral", "penguin", "goose", "duck", "eel",
                     "salamander", "newt", "cockroach", "mantis", "dragonfly",
                     "grasshopper", "ladybug", "tarantula", "trilobite",
                     "scorpion", "koi", "starfish", "urchin", "cucumber")),
    ("human_spaces", ("altar", "barn", "castle", "church", "cliff", "dam",
                      "fountain", "lakeside", "seashore", "obelisk", "palace",
                      "reef", "steel arch bridge", "suspension bridge", "viaduct",
                      "water tower", "beacon", "monastery", "triumphal arch",
                      "confectionery", "butcher shop", "cash machine")),
    ("vehicles", ("car", "truck", "bus", "van", "train", "locomotive", "boat",
                  "ship", "canoe", "gondola", "lifeboat", "tractor", "trolley",
                  "bicycle", "moped", "scooter", "limousine", "jinrikisha",
                  "wagon", "convertible", "cab", "go-kart", "sports car",
                  "school bus", "police van", "beach wagon")),
    ("food", ("pizza", "banana", "orange", "lemon", "pretzel", "bagel",
              "meat loaf", "guacamole", "ice cream", "ice lolly", "espresso",
              "potpie", "mashed potato", "cauliflower", "bell pepper",
              "broccoli", "mushroom", "acorn", "plate", "beer", "wine")),
)


def semantic_groups(
    wnids: list[str],
    labels: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    Assign every WordNet id to one of :data:`GROUPS`.

    Args:
        wnids:  the 200 TinyImageNet WordNet ids (from ``wnids.txt``).
        labels: optional ``wnid -> human-readable name`` map (from ``words.txt``),
                required only for the keyword fallback.

    Returns:
        ``{wnid: group_name}``. Anything unmatched lands in
        ``objects_tools_misc``, which is the paper's catch-all bucket.

    Prefers WordNet hypernym closure via NLTK. If NLTK or the ``wordnet`` corpus
    is missing, falls back to keyword matching on the readable label and prints a
    warning, because the fallback is coarser and will shift borderline classes.
    """
    try:
        return _groups_via_wordnet(wnids)
    except Exception as exc:  # nltk missing, corpus not downloaded, ...
        print(
            f"[splits] WordNet unavailable ({type(exc).__name__}: {exc}). "
            f"Falling back to keyword matching, which is coarser -- install nltk "
            f"and run `python -c \"import nltk; nltk.download('wordnet')\"` for "
            f"the intended grouping."
        )
        if labels is None:
            raise ValueError(
                "The keyword fallback needs human-readable labels. Pass "
                "labels=... loaded from TinyImageNet's words.txt."
            ) from exc
        return _groups_via_keywords(wnids, labels)


def _groups_via_wordnet(wnids: list[str]) -> dict[str, str]:
    """Group by WordNet hypernym closure. Raises if NLTK/WordNet is unavailable."""
    from nltk.corpus import wordnet as wn

    targets = {
        group: [wn.synset(name) for name in names]
        for group, names in _HYPERNYM_RULES
    }

    assignment: dict[str, str] = {}
    for wnid in wnids:
        # TinyImageNet wnids are "n" + 8-digit offset, e.g. n01443537.
        synset = wn.synset_from_pos_and_offset("n", int(wnid[1:]))
        closure = set(synset.closure(lambda s: s.hypernyms()))
        closure.add(synset)

        assignment[wnid] = "objects_tools_misc"
        for group, group_synsets in targets.items():
            if any(t in closure for t in group_synsets):
                assignment[wnid] = group
                break
    return assignment


def _groups_via_keywords(wnids: list[str], labels: dict[str, str]) -> dict[str, str]:
    """Group by substring match on the readable label. Safety net only."""
    assignment: dict[str, str] = {}
    for wnid in wnids:
        name = labels.get(wnid, "").lower()
        assignment[wnid] = "objects_tools_misc"
        for group, keywords in _KEYWORD_RULES:
            if any(kw in name for kw in keywords):
                assignment[wnid] = group
                break
    return assignment


def semantic_groups_from_csv(path: str | Path) -> dict[str, str]:
    """
    Load a previously derived grouping from CSV (columns: ``wnid``, ``group``).

    Committing the derived grouping is the way to make the split exactly
    reproducible across machines, since it removes any dependence on the
    installed WordNet version.
    """
    assignment: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            assignment[row["wnid"]] = row["group"]
    return assignment


def make_class_split(
    wnids: list[str],
    groups: dict[str, str],
    seed: int = 1,
) -> tuple[list[str], list[str]]:
    """
    Split classes into disjoint subsets A and B, balanced within semantic group.

    Within each group the member classes are sorted (so the result depends only
    on the seed, not on filesystem ordering) then shuffled with ``seed``, and the
    first half goes to A. Odd-sized groups alternate which side gets the extra
    class, so the leftovers do not all pile onto A and the totals stay at 100/100.

    Args:
        wnids:  all class ids to split.
        groups: ``{wnid: group_name}`` from :func:`semantic_groups`.
        seed:   the split seed. The paper's "split 1" is seed 1.

    Returns:
        ``(subset_a, subset_b)``, each a sorted list of wnids.
    """
    by_group: dict[str, list[str]] = defaultdict(list)
    for wnid in wnids:
        by_group[groups.get(wnid, "objects_tools_misc")].append(wnid)

    rng = random.Random(seed)
    subset_a: list[str] = []
    subset_b: list[str] = []

    # Alternate the odd-one-out between A and B, in a fixed group order so the
    # result is deterministic.
    odd_to_a = True
    for group in GROUPS:
        members = sorted(by_group.get(group, []))
        rng.shuffle(members)

        half = len(members) // 2
        if len(members) % 2 == 0:
            subset_a.extend(members[:half])
            subset_b.extend(members[half:])
        else:
            if odd_to_a:
                subset_a.extend(members[:half + 1])
                subset_b.extend(members[half + 1:])
            else:
                subset_a.extend(members[:half])
                subset_b.extend(members[half:])
            odd_to_a = not odd_to_a

    if len(subset_a) != len(subset_b):
        raise RuntimeError(
            f"Split is unbalanced: |A|={len(subset_a)}, |B|={len(subset_b)}. "
            f"This happens when the number of odd-sized semantic groups is odd; "
            f"check the grouping with describe_split()."
        )
    return sorted(subset_a), sorted(subset_b)


def describe_split(
    subset_a: list[str],
    subset_b: list[str],
    groups: dict[str, str],
) -> str:
    """
    Human-readable report of group balance, for comparing against Appendix A.1.

    Reports per-group A/B counts and flags any deviation from
    :data:`PAPER_SPLIT1_SIGNATURE`. A deviation indicates that the WordNet
    grouping resolved differently on this machine, and therefore that the split
    differs from the documented split 1; see the module-level note on pinning it.
    """
    set_a, set_b = set(subset_a), set(subset_b)
    lines = [
        f"Class split: |A| = {len(set_a)}, |B| = {len(set_b)}, "
        f"overlap = {len(set_a & set_b)}",
        f"{'group':<20} {'A':>4} {'B':>4} {'total':>6}",
    ]

    mismatches: list[str] = []
    for group in GROUPS:
        n_a = sum(1 for w in set_a if groups.get(w) == group)
        n_b = sum(1 for w in set_b if groups.get(w) == group)
        lines.append(f"{group:<20} {n_a:>4} {n_b:>4} {n_a + n_b:>6}")

        expected = PAPER_SPLIT1_SIGNATURE.get(group)
        if expected and (n_a, n_b) != expected:
            mismatches.append(
                f"  {group}: got {n_a}/{n_b}, paper split 1 reports "
                f"{expected[0]}/{expected[1]}"
            )
        elif expected is None and (n_a + n_b) % 2 == 0 and n_a != n_b:
            mismatches.append(
                f"  {group}: even-sized group ({n_a + n_b}) split unevenly as {n_a}/{n_b}"
            )

    if mismatches:
        lines.append("\nDeviations from the paper's documented split 1:")
        lines.extend(mismatches)
        lines.append(
            "  (Expected if the WordNet grouping differs; see splits.py caveat.)"
        )
    else:
        lines.append("\nMatches the documented split-1 signature.")

    return "\n".join(lines)

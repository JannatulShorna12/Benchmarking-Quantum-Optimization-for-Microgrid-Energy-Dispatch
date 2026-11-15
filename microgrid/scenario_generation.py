from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List

from sklearn.cluster import KMeans

from microgrid.config import N_SCENARIOS
from microgrid.preprocessing import ProcessedDay


@dataclass
class Scenario:
    """One k-means cluster representing a renewable scenario."""
    id:        int
    weight:    float      # fraction of members in this cluster
    solar:     np.ndarray  # shape (T,) - normalised solar forecast
    wind:      np.ndarray  # shape (T,) - normalised wind forecast
    n_members: int         # number of ECMWF members assigned to this cluster


ScenarioList = List[Scenario]


def generate_scenarios(
    pday: ProcessedDay,
    n_scenarios: int = N_SCENARIOS,
    random_state: int = 42,
    standardize_blocks: bool = False,
) -> ScenarioList:
    """Reduce the 51-member ECMWF ensemble to ``n_scenarios`` representative days.

    Clustering operates on the joint solar-wind trajectory of each ensemble member.
    Each member contributes one feature vector formed by concatenating its full-day
    solar and wind profiles at the 15-minute dispatch resolution, so the feature
    matrix is ``(51, 2 * N_STEPS)`` = (51, 192) - not a daily aggregate. Clustering on
    whole trajectories rather than daily totals keeps members with similar energy but
    different timing in separate scenarios, which is what matters for dispatch.

    Members are grouped by k-means under Euclidean distance, and each cluster is
    represented by its **medoid** - the real ensemble member closest to the cluster
    centroid - rather than by the centroid itself. Averaging members would smooth away
    the ramps and extremes that make a scenario binding.

    Both blocks are already in per-unit of ``P_scale``, so they share a common scale
    and enter the distance directly. Because this balancing authority is
    wind-dominated (solar is under 1% of load on every day of 2018), the unscaled
    distance is effectively driven by wind. Set ``standardize_blocks=True`` to
    z-score each block separately, giving solar and wind equal weight regardless of
    magnitude; this is reported as a sensitivity case rather than used by default,
    since it would weight a variable contributing under 1% of supply equally with one
    contributing up to 51%.

    Scenario probabilities are the empirical cluster frequencies, ``w_k = n_k / 51``.
    Returned scenarios are sorted by ascending mean renewable output, so index 0 is
    always the most conservative (lowest-supply) case.
    """
    n_members = pday.solar_ec_n.shape[0]   # 51

    # Each member -> [solar(96) | wind(96)] at 15-min resolution, per-unit of P_scale.
    features = np.hstack([pday.solar_ec_n, pday.wind_ec_n])  # (51, 192)

    if standardize_blocks:
        n_steps = pday.solar_ec_n.shape[1]
        features = features.copy()
        for lo, hi in ((0, n_steps), (n_steps, 2 * n_steps)):
            block = features[:, lo:hi]
            sd = block.std()
            if sd > 1e-12:
                features[:, lo:hi] = (block - block.mean()) / sd

    km     = KMeans(n_clusters=n_scenarios, random_state=random_state, n_init=20)
    labels = km.fit_predict(features)

    scenarios = []
    for k in range(n_scenarios):
        mask  = labels == k
        count = int(mask.sum())
        if count == 0:
            continue
        weight = count / n_members
        # Medoid: real ensemble member nearest to cluster centroid.
        # Preserves realistic extremes (centroid averaging smooths them out).
        cluster_features = features[mask]
        cluster_idxs     = np.where(mask)[0]
        dists            = np.linalg.norm(cluster_features - km.cluster_centers_[k], axis=1)
        medoid_local     = int(np.argmin(dists))
        medoid_member    = int(cluster_idxs[medoid_local])
        solar_k = pday.solar_ec_n[medoid_member]
        wind_k  = pday.wind_ec_n[medoid_member]
        scenarios.append(
            Scenario(id=k, weight=weight, solar=solar_k, wind=wind_k, n_members=count)
        )

    # Sort by ascending mean renewable (low -> high)
    scenarios.sort(key=lambda s: (s.solar + s.wind).mean())
    for i, s in enumerate(scenarios):
        s.id = i

    return scenarios


def scenarios_to_tuples(scenarios: ScenarioList):
    """Convert to the (weight, solar_arr, wind_arr) tuple format used by QUBO builders."""
    return [(s.weight, s.solar, s.wind) for s in scenarios]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Networked Social Simulation: Prisoner's Dilemma with Personality-Driven Tie Dynamics

Implements the model described in the primary paper:

- Dynamic network with personality-driven tie dynamics (Extraversion, Openness, Agreeableness).
- Controlled initial degrees: each node starts with degree in {1, 2}.
- Grid execution over combinations of parameter lists (N, beta, seeds, scenarios, etc.).
- Per-turn Spearman correlations (cumulative payoff vs metrics).
- Per-node clustering coefficient each turn.
- Per-turn assortativity (numeric attributes + degree).
- Parametric payoff matrix + symmetric per-interaction cost.
- Multi-seed aggregation with 95% CIs (per-turn + final summaries).
- Leniency δ for desertion; trait scenarios (baseline + skewed Betas per trait).
- Dyadic adaptation uses a Beta–Bernoulli posterior with prior strength s>0 centered at Agreeableness.
- CSVs are saved.

Dependencies: numpy, pandas, networkx, scipy

CITATION
--------

If you use this code or any datasets produced with it in scientific, academic or
technical work, you must cite the associated article and you are strongly
encouraged to also cite the software and dataset records, for example:

  Abián, D., Bernad, J., Ilarri, S. & Trillo-Lado, R. (2025).
  "Individual and collective gains from cooperation and reciprocity in a
   dynamic-network Prisoner’s Dilemma driven by extraversion, openness,
   and agreeableness."
  Journal / preprint server, volume(issue), pages. DOI: TODO_INSERT_ARTICLE_DOI

  Abián, D. (2025).
  "bigthree-dynet-pd: Dynamic-network Prisoner's Dilemma simulation with
   personality-driven tie dynamics" [Computer software].
  Zenodo. DOI: TODO_INSERT_SOFTWARE_DOI

  Abián, D. (2025).
  "Simulation outputs for 'Individual and collective gains from cooperation
   and reciprocity in a dynamic-network Prisoner’s Dilemma driven by
   extraversion, openness, and agreeableness'" [Data set].
  Zenodo. DOI: https://doi.org/10.5281/zenodo.17714612

See README.md and CITATION.cff for up-to-date citation details.
"""

import argparse
import itertools
import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Any, Iterable, Optional

import numpy as np
import scipy
import pandas as pd
import networkx as nx
import warnings
from scipy.stats import spearmanr, t, ConstantInputWarning


# =========================
# Agent and Simulation Core
# =========================

@dataclass
class Agent:
    idx: int
    extraversion: float
    openness: float
    agreeableness: float

    def ideal_degree(self, min_deg: int, max_deg: int) -> float:
        return min_deg + self.extraversion * (max_deg - min_deg)


class NetworkPDSim:
    """
    Prisoner's Dilemma on a dynamic network with personality-driven tie dynamics.

    Key components (matching the Methods section):

    • Ideal degree (Extraversion):
        D*_i = d_min + E_i (d_max − d_min)

    • Candidate pools and Openness:
        For each agent i at turn t, with neighbours N_i(t),

            FoF_i(t) = {ℓ : ∃ j in N_i(t) such that ℓ in N_j(t),
                             ℓ not in N_i(t), ℓ != i}
            Out_i(t) = (V minus (N_i(t) ∪ {i})) minus FoF_i(t)
            m        = |FoF_i(t)|
            M        = |Out_i(t)|

        Openness O_i determines the number of outsiders u_i(t) added to the
        discovery pool via:

            u_i(t) =
                0,                                              if M = 0
                [ ceil( O_i/(1-O_i) * m ) ]_1^{ceil(M/2)},      if M>0, m>0, O_i < 1
                ceil( 0.5 * O_i * M ),                          otherwise (m = 0 or O_i = 1)

        where [·]_1^{ceil(M/2)} clips to [1, ceil(M/2)].
        The discovery pool is FoF_i(t) ∪ sample(Out_i(t), u_i(t)), with sampling
        without replacement.

    • Discovery is unilateral; any pair {i,j} is eligible if i lists j or j lists i.

    • New tie formation:
        p_add(i,j,t) = clamp(
            λ * ( (D*_i − k_i(t))/L_i(t) + (D*_j − k_j(t))/L_j(t) ) / 2,
            0, 1
        )

        where L_i(t) is the number of candidate pairs incident on i and λ is
        the global damping factor.

    • Desertion (tie cutting):
        For each directed dyad i ← j we maintain counts of j’s behaviour toward i:
            n^C_{j→i}, n^D_{j→i}

        A defection-risk score uses the payoff-asymmetry factor η:

            ω_{i←j}(t) = η n^D_{j→i}(t) / (η n^D_{j→i}(t) + n^C_{j→i}(t) + 1)

        where η = |S − c| / (R − c), with (R,S) from the PD payoffs and c the
        symmetric per-edge cost. Leniency δ defines a soft ideal degree
        (1 − δ) D*_i, and cutting pressure is:

            excess_i(t) = λ ( k_i(t) − (1 − δ) D*_i )_+

        Conditional on excess_i(t) and ω_{i←j}(t), the probability that i deserts j is:

            p_cut(i → j, t) = clamp(
                excess_i(t) * ω_{i←j}(t) / Σ_{ℓ in N_i(t)} ω_{i←ℓ}(t),
                0, 1
            )

    • Cooperation rule (Agreeableness + dyadic history):
        Directed dyadic counts (n^C_{j→i}, n^D_{j→i}) define a Beta–Bernoulli
        posterior with prior strength s > 0 and prior mean A_i (Agreeableness):

            post_mean_{i←j}(t) =
                ( n^C_{j→i}(t) + s A_i ) / ( n^C_{j→i}(t) + n^D_{j→i}(t) + s )

        and cooperation probability:

            p(C_i | j, t) = β A_i + (1 − β) post_mean_{i←j}(t)

        with β in [0,1]. With no history, the first move follows disposition:
        p(C_i | j, t) = A_i for any β.

    • Directed dyadic counters persist across cuts, and there is no global
      reputation or third-party punishment.
    """

    CORR_KEYS = (
        "extraversion", "openness", "agreeableness",
        "coops_done", "defs_done",
        "cuts_initiated", "cuts_suffered",
        "coops_received", "defs_received",
        "degree",
    )

    def __init__(
        self,
        n_agents: int = 80,
        n_turns: int = 100,
        seed: int = 42,
        min_ideal_degree: int = 1,
        max_ideal_degree: int = 10,
        beta: float = 0.5,
        init_graph: str = "bounded12",  # "bounded12" or "er"
        init_degree2_fraction: float = 0.5,  # if bounded12
        init_edge_prob: float = 0.05,        # if er
        personality_source: str = "beta",    # "beta" or "truncnorm"
        personality_params: Dict[str, Dict[str, float]] = None,
        payoff_R: float = 3.0, payoff_P: float = 1.0,
        payoff_T: float = 5.0, payoff_S: float = 0.0,
        exchange_cost: float = 2.0,
        output_dir: str = None,
        make_run_plots: bool = False,        # deprecated; retained for backward compat
        save_run_network_stats: bool = True,
        leniency_delta: float = 0.20,        # desertion leniency δ
        trait_scenario: str = "baseline",    # record-only; affects sampling upstream
        prior_strength_s: float = 0.5,       # Beta–Bernoulli prior strength s
        damping_lambda: float = 0.5,         # global λ in [0,1], used in add & cut
    ):
        self.n_agents = int(n_agents)
        self.n_turns = int(n_turns)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        # Keep global RNG seeding for backwards-compatibility / reproducibility
        np.random.seed(self.seed)

        self.min_ideal_degree = int(min_ideal_degree)
        self.max_ideal_degree = int(max_ideal_degree)
        if self.max_ideal_degree < self.min_ideal_degree:
            raise ValueError("max_ideal_degree must be >= min_ideal_degree.")

        self.beta = float(np.clip(beta, 0.0, 1.0))
        self.init_graph = str(init_graph)
        self.init_degree2_fraction = float(np.clip(init_degree2_fraction, 0.0, 1.0))
        self.init_edge_prob = float(np.clip(init_edge_prob, 0.0, 1.0))
        self.personality_source = str(personality_source)
        self.leniency_delta = float(max(0.0, min(1.0, leniency_delta)))  # clamp to [0,1]
        self.trait_scenario = str(trait_scenario)
        self.prior_strength_s = float(prior_strength_s)
        self.damping_lambda = float(np.clip(damping_lambda, 0.0, 1.0))

        if self.n_agents < 2:
            raise ValueError("n_agents must be at least 2.")
        if self.n_turns < 1:
            raise ValueError("n_turns must be at least 1.")
        if self.prior_strength_s <= 0.0:
            raise ValueError("prior_strength_s must be strictly positive.")

        if personality_params is None:
            if personality_source == "beta":
                personality_params = {
                    "extraversion": {"a": 5.0, "b": 5.0},
                    "openness": {"a": 5.0, "b": 5.0},
                    "agreeableness": {"a": 5.0, "b": 5.0},
                }
            else:
                personality_params = {"mean": 0.5, "sd": 0.15}
        self.personality_params = personality_params

        # PD payoffs and symmetric per-edge exchange cost
        self.payoff_R = float(payoff_R)
        self.payoff_P = float(payoff_P)
        self.payoff_T = float(payoff_T)
        self.payoff_S = float(payoff_S)
        self.exchange_cost = float(exchange_cost)

        # Prisoner's Dilemma sanity check: T > R > P > S
        if not (self.payoff_T > self.payoff_R > self.payoff_P > self.payoff_S):
            raise ValueError(
                f"Payoff matrix must satisfy T>R>P>S (Prisoner's Dilemma). "
                f"Got T={self.payoff_T}, R={self.payoff_R}, "
                f"P={self.payoff_P}, S={self.payoff_S}."
            )

        # Net one-step payoffs (before desertion weighting)
        net_R = self.payoff_R - self.exchange_cost
        net_S = self.payoff_S - self.exchange_cost

        # In the paper, mutual cooperation is a net gain and being exploited is a net loss.
        if net_R <= 0.0 or net_S >= 0.0:
            warnings.warn(
                "With the provided payoffs and exchange_cost, mutual cooperation may not be "
                "net positive and/or being exploited may not be net negative. "
                "This deviates from the configuration used in the paper "
                "(R - c > 0 and S - c < 0).",
                RuntimeWarning,
            )

        # Payoff-asymmetry factor η = |S − c| / (R − c), used in desertion.
        if net_R != 0.0:
            self.payoff_eta = abs(net_S) / net_R
        else:
            # Fallback: treat defections and cooperations symmetrically in desertion.
            self.payoff_eta = 1.0
        assert self.payoff_eta >= 0.0, "Payoff asymmetry factor η must be non-negative."

        # Output dir
        if output_dir is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            output_dir = f"./network_pd_run_{ts}"
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        # Per-run output controls (no plots produced, network stats CSV optional)
        self.make_run_plots = False
        self.save_run_network_stats = bool(save_run_network_stats)

        # Graph and storage
        self.G = nx.Graph()
        self.agents: List[Agent] = []

        # Directed dyadic counters (persistent)
        self.coop_ij = np.zeros((self.n_agents, self.n_agents), dtype=np.int64)
        self.def_ij = np.zeros((self.n_agents, self.n_agents), dtype=np.int64)

        # Per-agent cumulative metrics (cumulative payoff == "points")
        self.points = np.zeros(self.n_agents, dtype=np.float64)
        self.coops_done = np.zeros(self.n_agents, dtype=np.int64)
        self.defs_done = np.zeros(self.n_agents, dtype=np.int64)
        self.coops_received = np.zeros(self.n_agents, dtype=np.int64)
        self.defs_received = np.zeros(self.n_agents, dtype=np.int64)
        self.cuts_initiated = np.zeros(self.n_agents, dtype=np.int64)
        self.cuts_suffered = np.zeros(self.n_agents, dtype=np.int64)

        # Logs
        self.agent_timeseries_records: List[Dict] = []
        self.corr_records: List[Dict] = []       # cumulative payoff vs metrics
        self.assort_records: List[Dict] = []     # assortativity over time
        self.network_records: List[Dict] = []

        # Init
        self._init_agents()
        self._init_network()

        # No per-run figures
        self.atlas_start_path = None

        # Sanity check to match paper: 2R > T + S
        if not (2 * self.payoff_R > self.payoff_T + self.payoff_S):
            raise ValueError("Payoffs should satisfy 2R > T + S (paper setting).")


    # -----------------------------
    # Initialization
    # -----------------------------

    def _sample_trait_beta(self, a: float, b: float, size: int) -> np.ndarray:
        return self.rng.beta(a, b, size=size)

    def _sample_trait_truncnorm(self, mean: float, sd: float, size: int) -> np.ndarray:
        out = []
        while len(out) < size:
            draw = self.rng.normal(mean, sd)
            if 0.0 <= draw <= 1.0:
                out.append(draw)
        return np.array(out)

    def _init_agents(self):
        if self.personality_source == "beta":
            e = self._sample_trait_beta(**self.personality_params["extraversion"], size=self.n_agents)
            o = self._sample_trait_beta(**self.personality_params["openness"], size=self.n_agents)
            a = self._sample_trait_beta(**self.personality_params["agreeableness"], size=self.n_agents)
        else:
            mean = float(self.personality_params.get("mean", 0.5))
            sd = float(self.personality_params.get("sd", 0.15))
            e = self._sample_trait_truncnorm(mean, sd, self.n_agents)
            o = self._sample_trait_truncnorm(mean, sd, self.n_agents)
            a = self._sample_trait_truncnorm(mean, sd, self.n_agents)

        assert len(e) == self.n_agents and len(o) == self.n_agents and len(a) == self.n_agents
        for arr in (e, o, a):
            assert np.all((arr >= 0.0) & (arr <= 1.0)), "Trait draws must be in [0,1]."

        self.agents = [
            Agent(i, float(e[i]), float(o[i]), float(a[i]))
            for i in range(self.n_agents)
        ]
        for ag in self.agents:
            self.G.add_node(
                ag.idx,
                extraversion=ag.extraversion,
                openness=ag.openness,
                agreeableness=ag.agreeableness,
                ideal_degree=ag.ideal_degree(self.min_ideal_degree, self.max_ideal_degree),
            )

        # Invariants: traits and ideal degrees
        assert len(self.agents) == self.n_agents, "Number of agents must match n_agents."
        for ag in self.agents:
            ideal = ag.ideal_degree(self.min_ideal_degree, self.max_ideal_degree)
            assert self.min_ideal_degree <= ideal <= self.max_ideal_degree, \
                "Ideal degree out of configured range."

        assert self.G.number_of_nodes() == self.n_agents, "Graph must contain all agents."


    def _init_network(self):
        if self.init_graph == "er":
            for i in range(self.n_agents):
                for j in range(i + 1, self.n_agents):
                    if self.rng.random() < self.init_edge_prob:
                        self.G.add_edge(i, j)
        else:
            self._init_network_bounded12(self.init_degree2_fraction)

    def _init_network_bounded12(self, frac_degree2: float):
        n = self.n_agents
        if n < 2:
            raise ValueError("bounded12 init requires n_agents >= 2")

        nodes = list(range(n))
        self.rng.shuffle(nodes)
        # Ensure degree >= 1
        for k in range(0, n - 1, 2):
            u, v = nodes[k], nodes[k + 1]
            self.G.add_edge(u, v)

        if n % 2 == 1:
            leftover = nodes[-1]
            candidates = [i for i in range(n) if i != leftover and not self.G.has_edge(i, leftover)]
            if not candidates:
                candidates = [i for i in range(n) if i != leftover]
            partner = int(self.rng.choice(candidates))
            self.G.add_edge(leftover, partner)

        current_deg2 = int(np.sum([self.G.degree(i) == 2 for i in range(n)]))
        target_deg2 = int(round(frac_degree2 * n))
        if target_deg2 <= current_deg2:
            # Already at or above requested fraction of degree-2 nodes
            pass
        else:
            needed = target_deg2 - current_deg2
            if needed % 2 == 1:
                needed -= 1

            def deg1_nodes():
                return [i for i in range(n) if self.G.degree(i) == 1]

            attempts, max_attempts = 0, n * n
            while needed > 0 and attempts < max_attempts:
                d1 = deg1_nodes()
                if len(d1) < 2:
                    break
                self.rng.shuffle(d1)
                u = d1[0]
                v = next((c for c in d1[1:] if not self.G.has_edge(u, c)), None)
                if v is None:
                    attempts += 1
                    continue
                self.G.add_edge(u, v)
                needed -= 2
                attempts += 1

        # Final invariant: degrees in {1, 2}
        for i in range(n):
            deg_i = self.G.degree(i)
            if not (1 <= deg_i <= 2):
                raise RuntimeError("Initialization failed to enforce degree in {1,2}")
        
        assert self.G.number_of_nodes() == n, "Init graph lost or gained nodes unexpectedly."


    # -----------------------------
    # Helpers
    # -----------------------------

    def _neighbors_set(self) -> List[Set[int]]:
        return [set(self.G.neighbors(i)) for i in range(self.n_agents)]

    def _degree_array(self) -> np.ndarray:
        return np.array([self.G.degree(i) for i in range(self.n_agents)], dtype=np.int64)

    # -----------------------------
    # Candidate Pools & New Edges
    # -----------------------------

    def _build_candidate_pools(self, neigh: List[Set[int]]) -> List[Set[int]]:
        """
        Candidate pools (Methods).

        For each agent i at turn t, with neighbours N_i(t):

            FoF_i(t)   = {ℓ : ∃ j in N_i(t) such that ℓ in N_j(t),
                                ℓ not in N_i(t), ℓ != i}
            Out_i(t)   = all nodes in V that are neither neighbours of i,
                         nor i itself, nor friends-of-friends of i
                         (i.e., nodes outside N_i(t) ∪ {i} ∪ FoF_i(t))
            m          = |FoF_i(t)|
            M          = |Out_i(t)|

        Openness O_i determines how many outsiders u_i(t) are added:

            u_i(t) =
                0,                                              if M = 0
                [ ceil( O_i/(1-O_i) * m ) ]_1^{ceil(M/2)},      if M>0, m>0, O_i < 1
                ceil( 0.5 * O_i * M ),                          otherwise  (m = 0 or O_i = 1)

        where [·]_1^{ceil(M/2)} clips to the interval [1, ceil(M/2)].

        At most half of all outsiders are queried in a turn.
        """
        all_nodes = set(range(self.n_agents))
        pools: List[Set[int]] = [set() for _ in range(self.n_agents)]

        for i in range(self.n_agents):
            nbh_i = neigh[i]

            # Friends-of-friends
            fof = set()
            for v in nbh_i:
                fof.update(neigh[v])
            fof.discard(i)
            fof -= nbh_i

            # Outsiders
            non_neighbors = all_nodes - nbh_i - {i}
            outsiders = list(non_neighbors - fof)

            m = len(fof)
            M = len(outsiders)
            assert m >= 0 and M >= 0

            if M == 0:
                pools[i] = fof
                continue

            Oi = float(np.clip(self.agents[i].openness, 0.0, 1.0))

            if M > 0 and m > 0 and Oi < 1.0:
                # clamp( ceil( O_i/(1−O_i) * m ), 1, ceil(M/2) )
                if Oi == 0.0:
                    # avoid division by zero, target = 0 then clipped to 1
                    target = 0
                else:
                    target = int(math.ceil((Oi / (1.0 - Oi)) * m))
                upper = int(math.ceil(M / 2.0))
                u = max(1, min(upper, target))
            else:
                # Edge case: m = 0 or O_i = 1 -> ceil(0.5 * O_i * M)
                u = int(math.ceil(0.5 * Oi * M))

            # Final safety: 0 <= u <= M
            u = max(0, min(M, u))

            pool = set(fof)
            if u > 0:
                chosen = self.rng.choice(outsiders, size=u, replace=False)
                pool.update(map(int, np.atleast_1d(chosen).tolist()))
            
            # --- Invariants for candidate pool of i ---
            # 1) i is never in its own pool
            assert i not in pool, "Self appeared in own candidate pool."
            # 2) No current neighbours in the pool
            assert pool.isdisjoint(nbh_i), "Neighbour found in candidate pool."
            # 3) All nodes are valid indices
            assert all(0 <= j < self.n_agents for j in pool), "Invalid node index in pool."

            pools[i] = pool

        return pools

    def _try_add_new_edges(self, pools: List[Set[int]], degrees: np.ndarray):
        """
        Build C(t) = { {i,j} : j in P_i(t) or i in P_j(t) }, then
        L_i(t) = |{ ℓ ≠ i : {i,ℓ} in C(t) }|.

        p_add(i,j,t) = clamp(
            λ * ( (D*_i - k_i(t))/L_i(t) + (D*_j - k_j(t))/L_j(t) ) / 2,
            0, 1
        )

        If L_x == 0, that term contributes 0. Signed gaps are allowed; net
        negative pressure is clamped to 0 probability.
        """
        m_before = self.G.number_of_edges()
        ideals = np.array(
            [self.G.nodes[i]["ideal_degree"] for i in range(self.n_agents)],
            dtype=np.float64
        )

        # Candidate pairs (unilateral discovery)
        candidate_pairs: Set[Tuple[int, int]] = set()
        for i in range(self.n_agents):
            for j in pools[i]:
                if i != j and not self.G.has_edge(i, j):
                    candidate_pairs.add((min(i, j), max(i, j)))
        for j in range(self.n_agents):
            for i in pools[j]:
                if i != j and not self.G.has_edge(i, j):
                    candidate_pairs.add((min(i, j), max(i, j)))

        # L_i(t): incident candidate count
        L = np.zeros(self.n_agents, dtype=np.int64)
        for (i, j) in candidate_pairs:
            L[i] += 1
            L[j] += 1
        L = L.astype(float)

        lam = self.damping_lambda
        for (i, j) in candidate_pairs:
            assert i != j, "Candidate pair with self-loop."
            Li = L[i]
            Lj = L[j]
            term_i = 0.0 if Li <= 0.0 else (ideals[i] - degrees[i]) / Li
            term_j = 0.0 if Lj <= 0.0 else (ideals[j] - degrees[j]) / Lj
            raw = lam * 0.5 * (term_i + term_j)
            p = float(np.clip(raw, 0.0, 1.0))
            if self.rng.random() < p:
                # Graph is simple; NetworkX forbids parallel edges.
                self.G.add_edge(i, j)
        
        m_after = self.G.number_of_edges()
        # Edge addition should only increase or keep the number of edges
        assert m_after >= m_before, "Edge addition step unexpectedly removed edges."

    # -----------------------------
    # Desertion (Cuts)
    # -----------------------------

    def _desert_decisions(self, neigh: List[Set[int]], degrees: np.ndarray):
        """
        Desertion decisions, matching Eq. (pcut) in the Methods section.

        For each i, define the soft ideal degree:

            soft_ideal_i = (1 − δ) D*_i

        and the excess degree:

            excess_i(t) = λ ( k_i(t) − soft_ideal_i )_+

        The defection-risk score uses the payoff-asymmetry factor η:

            ω_{i←j}(t) = η n^D_{j→i}(t) / (η n^D_{j→i}(t) + n^C_{j→i}(t) + 1)

        Then:

            p_cut(i → j, t) = clamp(
                excess_i(t) * ω_{i←j}(t) / Σ_{ℓ in N_i(t)} ω_{i←ℓ}(t),
                0, 1
            )
        """
        ideals = np.array(
            [self.G.nodes[i]["ideal_degree"] for i in range(self.n_agents)],
            dtype=np.float64
        )

        eta = float(self.payoff_eta)
        assert eta >= 0.0

        r_mats: Dict[int, Dict[int, float]] = {}
        sum_r = np.zeros(self.n_agents, dtype=np.float64)
        for i in range(self.n_agents):
            total = 0.0
            r_i: Dict[int, float] = {}
            for j in neigh[i]:
                defs_on_i = self.def_ij[j, i]
                coops_on_i = self.coop_ij[j, i]
                r = (eta * float(defs_on_i)) / (eta * float(defs_on_i) + float(coops_on_i) + 1.0)
                r_i[j] = r
                total += r
                assert 0.0 <= r <= 1.0, "Defection-risk score r must be in [0,1]."
            r_mats[i] = r_i
            sum_r[i] = total
            assert sum_r[i] >= 0.0, "Sum of risk scores must be non-negative."

        edges_to_remove: List[Tuple[int, int]] = []
        deserter_flags: Dict[Tuple[int, int], Tuple[bool, bool]] = {}

        for i in range(self.n_agents):
            if not neigh[i]:
                continue
            soft_ideal = (1.0 - self.leniency_delta) * ideals[i]
            excess_raw = degrees[i] - soft_ideal
            excess = self.damping_lambda * (excess_raw if excess_raw > 0.0 else 0.0)
            denom = sum_r[i]
            for j in neigh[i]:
                if denom <= 0.0 or excess <= 0.0:
                    p_desert = 0.0
                else:
                    p_desert = excess * (r_mats[i][j] / denom)
                p_desert = float(np.clip(p_desert, 0.0, 1.0))
                assert 0.0 <= p_desert <= 1.0, "Desertion probability must be in [0,1]."
                will_desert = self.rng.random() < p_desert
                key = (min(i, j), max(i, j))
                if key not in deserter_flags:
                    deserter_flags[key] = (False, False)
                deserter_flags[key] = (
                    will_desert if i < j else deserter_flags[key][0],
                    deserter_flags[key][1] if i < j else will_desert,
                )

        for (i, j), (dij, dji) in deserter_flags.items():
            if dij or dji:
                edges_to_remove.append((i, j))
                if dij:
                    self.cuts_initiated[i] += 1
                    self.cuts_suffered[j] += 1
                if dji:
                    self.cuts_initiated[j] += 1
                    self.cuts_suffered[i] += 1

        for (i, j) in edges_to_remove:
            if self.G.has_edge(i, j):
                self.G.remove_edge(i, j)

        # Degrees must remain non-negative and consistent with edge count
        degrees_after = np.array([self.G.degree(i) for i in range(self.n_agents)], dtype=int)
        assert np.all(degrees_after >= 0), "Negative degree detected after desertion."
        assert degrees_after.sum() == 2 * self.G.number_of_edges(), \
            "Sum of degrees must equal 2 * number_of_edges after desertion."

    # -----------------------------
    # Prisoner's Dilemma
    # -----------------------------

    def _play_pd_on_remaining_edges(self, neigh: List[Set[int]]):
        """
        Play a one-shot Prisoner's Dilemma on each undirected edge, using the
        cooperation rule that blends Agreeableness and dyadic Beta–Bernoulli
        posteriors, as described in the Methods section.
        """
        coop_decision: Dict[Tuple[int, int], bool] = {}
        s = self.prior_strength_s  # prior strength in the Beta–Bernoulli smoothing

        for i in range(self.n_agents):
            Ai = float(self.agents[i].agreeableness)
            assert 0.0 <= Ai <= 1.0
            for j in neigh[i]:
                if (i, j) in coop_decision:
                    continue

                Aj = float(self.agents[j].agreeableness)
                assert 0.0 <= Aj <= 1.0

                # History of partner j toward ego i (up to t-1)
                c_ji = float(self.coop_ij[j, i])
                d_ji = float(self.def_ij[j, i])
                post_mean_ij = (c_ji + s * Ai) / (c_ji + d_ji + s)

                p_ij = float(np.clip(self.beta * Ai + (1.0 - self.beta) * post_mean_ij, 0.0, 1.0))

                # History of partner i toward ego j (up to t-1)
                c_ij = float(self.coop_ij[i, j])
                d_ij = float(self.def_ij[i, j])
                post_mean_ji = (c_ij + s * Aj) / (c_ij + d_ij + s)

                p_ji = float(np.clip(self.beta * Aj + (1.0 - self.beta) * post_mean_ji, 0.0, 1.0))

                # --- Invariants for cooperation probabilities ---
                assert c_ji >= 0 and d_ji >= 0 and c_ij >= 0 and d_ij >= 0, \
                    "History counts must be non-negative."
                assert c_ji + d_ji + s > 0 and c_ij + d_ij + s > 0, \
                    "Posterior denominator must be positive."
                assert 0.0 <= post_mean_ij <= 1.0 and 0.0 <= post_mean_ji <= 1.0, \
                    "Posterior mean must be in [0,1]."
                assert 0.0 <= p_ij <= 1.0 and 0.0 <= p_ji <= 1.0, \
                    "Cooperation probabilities must be in [0,1]."

                # Independent draws
                coop_decision[(i, j)] = self.rng.random() < p_ij
                coop_decision[(j, i)] = self.rng.random() < p_ji

        # Apply decisions, update dyadic counters and payoffs
        for i in range(self.n_agents):
            for j in list(neigh[i]):
                if i < j and self.G.has_edge(i, j):
                    assert (i, j) in coop_decision and (j, i) in coop_decision, \
                        "Missing coop_decision for an existing edge."

                    ci = coop_decision[(i, j)]
                    cj = coop_decision[(j, i)]

                    if ci:
                        self.coop_ij[i, j] += 1
                        self.coops_done[i] += 1
                        self.coops_received[j] += 1
                    else:
                        self.def_ij[i, j] += 1
                        self.defs_done[i] += 1
                        self.defs_received[j] += 1

                    if cj:
                        self.coop_ij[j, i] += 1
                        self.coops_done[j] += 1
                        self.coops_received[i] += 1
                    else:
                        self.def_ij[j, i] += 1
                        self.defs_done[j] += 1
                        self.defs_received[i] += 1

                    # Parametric payoffs (before per-edge cost)
                    if ci and cj:
                        self.points[i] += self.payoff_R
                        self.points[j] += self.payoff_R
                    elif (not ci) and (not cj):
                        self.points[i] += self.payoff_P
                        self.points[j] += self.payoff_P
                    elif ci and (not cj):
                        self.points[i] += self.payoff_S
                        self.points[j] += self.payoff_T
                    else:
                        self.points[i] += self.payoff_T
                        self.points[j] += self.payoff_S

                    # Symmetric exchange cost per active edge per turn
                    self.points[i] -= self.exchange_cost
                    self.points[j] -= self.exchange_cost

    # -----------------------------
    # Logging, Correlations, Assortativity
    # -----------------------------

    def _metric_arrays(self, degrees: np.ndarray) -> Dict[str, np.ndarray]:
        extraversion = np.array([a.extraversion for a in self.agents])
        openness = np.array([a.openness for a in self.agents])
        agreeableness = np.array([a.agreeableness for a in self.agents])
        return {
            "extraversion": extraversion,
            "openness": openness,
            "agreeableness": agreeableness,
            "coops_done": self.coops_done.astype(float),
            "defs_done": self.defs_done.astype(float),
            "cuts_initiated": self.cuts_initiated.astype(float),
            "cuts_suffered": self.cuts_suffered.astype(float),
            "coops_received": self.coops_received.astype(float),
            "defs_received": self.defs_received.astype(float),
            "degree": np.array(degrees, dtype=float),
        }

    @staticmethod
    def _safe_spearman(x: np.ndarray, y: np.ndarray):
        x = np.asarray(x)
        y = np.asarray(y)
        m = np.isfinite(x) & np.isfinite(y)
        n = int(m.sum())
        if n < 2:
            return float("nan"), 1.0, n
        if np.nanstd(x[m]) == 0.0 or np.nanstd(y[m]) == 0.0:
            return float("nan"), 1.0, n
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            rho, p = spearmanr(x[m], y[m])
        return float(rho), float(p), n

    def _log_agent_timeseries(self, turn_idx: int, degrees: np.ndarray):
        clustering = nx.clustering(self.G)
        for i in range(self.n_agents):
            self.agent_timeseries_records.append(
                dict(
                    turn=turn_idx,
                    agent=i,
                    points=float(self.points[i]),
                    degree=int(degrees[i]),
                    extraversion=float(self.agents[i].extraversion),
                    openness=float(self.agents[i].openness),
                    agreeableness=float(self.agents[i].agreeableness),
                    ideal_degree=float(self.G.nodes[i]["ideal_degree"]),
                    coops_done=int(self.coops_done[i]),
                    defs_done=int(self.defs_done[i]),
                    coops_received=int(self.coops_received[i]),
                    defs_received=int(self.defs_received[i]),
                    cuts_initiated=int(self.cuts_initiated[i]),
                    cuts_suffered=int(self.cuts_suffered[i]),
                    clustering=float(clustering.get(i, 0.0)),
                )
            )

    def _log_correlations_points(self, turn_idx: int, degrees: np.ndarray):
        pts = self.points.copy()
        for name, arr in self._metric_arrays(degrees).items():
            rho, p, n = self._safe_spearman(pts, arr)
            self.corr_records.append(
                {"turn": turn_idx, "variable": name, "rho": rho, "p_value": p, "n": n}
            )

    def _log_assortativity(self, turn_idx: int, degrees: np.ndarray):
        vars_now = self._metric_arrays(degrees)

        # Attach node attributes for numeric assortativity (everything excepto "degree")
        for name, arr in vars_now.items():
            if name != "degree":
                nx.set_node_attributes(
                    self.G,
                    {i: float(arr[i]) for i in range(self.n_agents)},
                    name,
                )

        m_edges = self.G.number_of_edges()

        for name, arr in vars_now.items():
            try:
                # Si no hay aristas, no hay assortativity que calcular
                if m_edges == 0:
                    assort = float("nan")
                else:
                    # Evitar casos en los que el atributo es constante
                    values = np.asarray(arr, dtype=float)
                    if np.all(~np.isfinite(values)) or np.nanstd(values) == 0.0:
                        assort = float("nan")
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            if name == "degree":
                                assort = float(nx.degree_assortativity_coefficient(self.G))
                            else:
                                assort = float(
                                    nx.numeric_assortativity_coefficient(self.G, name)
                                )
            except Exception:
                assort = float("nan")
            
            if not math.isnan(assort):
                assert -1.0000001 <= assort <= 1.0000001, \
                    "Assortativity must lie in [-1,1]."

            self.assort_records.append(
                {"turn": turn_idx, "variable": name, "assortativity": assort}
            )

    def _log_network_stats(self, turn_idx: int):
        n = self.n_agents
        m = self.G.number_of_edges()
        avg_degree = 2.0 * m / n if n > 0 else float("nan")
        self.network_records.append(
            dict(
                turn=turn_idx,
                num_edges=m,
                avg_degree=avg_degree,
                density=nx.density(self.G),
                avg_clustering=float(nx.average_clustering(self.G))
                if self.G.number_of_edges()
                else float("nan"),
            )
        )

    # -----------------------------
    # Simulation loop
    # -----------------------------

    def run(self):
        for t in range(1, self.n_turns + 1):
            degrees = self._degree_array()
            neigh = self._neighbors_set()

            pools = self._build_candidate_pools(neigh)
            self._try_add_new_edges(pools, degrees)

            degrees = self._degree_array()
            neigh = self._neighbors_set()
            self._desert_decisions(neigh, degrees)

            neigh = self._neighbors_set()
            self._play_pd_on_remaining_edges(neigh)

            degrees = self._degree_array()

            # --- Per-turn structural invariants ---
            assert self.G.number_of_nodes() == self.n_agents, \
                "Number of nodes changed during simulation."
            assert degrees.sum() == 2 * self.G.number_of_edges(), \
                "Degree sum must be 2 * number_of_edges each turn."

            self._log_agent_timeseries(t, degrees)
            self._log_correlations_points(t, degrees)
            self._log_assortativity(t, degrees)
            self._log_network_stats(t)

    # -----------------------------
    # Persistence
    # -----------------------------

    def save_all(self) -> Dict[str, Any]:
        # No per-run atlas
        atlas_end_path = None

        # Parameters
        params = {
            "n_agents": self.n_agents,
            "n_turns": self.n_turns,
            "seed": self.seed,
            "min_ideal_degree": self.min_ideal_degree,
            "max_ideal_degree": self.max_ideal_degree,
            "beta": self.beta,
            "init_graph": self.init_graph,
            "init_degree2_fraction": self.init_degree2_fraction,
            "init_edge_prob": self.init_edge_prob,
            "personality_source": self.personality_source,
            "personality_params": self.personality_params,
            "payoffs": {
                "R": self.payoff_R,
                "P": self.payoff_P,
                "T": self.payoff_T,
                "S": self.payoff_S,
                "exchange_cost": self.exchange_cost,
                "payoff_asymmetry_eta": self.payoff_eta,
            },
            "leniency_delta": self.leniency_delta,
            "damping_lambda": self.damping_lambda,
            "trait_scenario": self.trait_scenario,
            "dyadic_prior_strength_s": self.prior_strength_s,
            "notes": "All plotting disabled; only CSV outputs are produced.",
            "library_versions": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "networkx": nx.__version__,
                "scipy": scipy.__version__,
            },
        }
        with open(os.path.join(self.output_dir, "parameters.json"), "w") as f:
            json.dump(params, f, indent=2)

        # Agents (static)
        agents_df = pd.DataFrame(
            {
                "agent": [a.idx for a in self.agents],
                "extraversion": [a.extraversion for a in self.agents],
                "openness": [a.openness for a in self.agents],
                "agreeableness": [a.agreeableness for a in self.agents],
                "ideal_degree": [self.G.nodes[a.idx]["ideal_degree"] for a in self.agents],
            }
        )
        agents_csv = os.path.join(self.output_dir, "agents.csv")
        agents_df.to_csv(agents_csv, index=False)

        # Time series per agent
        ts_df = pd.DataFrame(self.agent_timeseries_records)
        ts_csv = os.path.join(self.output_dir, "agent_timeseries.csv")
        ts_df.to_csv(ts_csv, index=False)

        # Correlations (cumulative payoff vs metrics)
        corr_df = pd.DataFrame(self.corr_records)
        corr_csv = os.path.join(self.output_dir, "correlations.csv")
        corr_df.to_csv(corr_csv, index=False)

        # Assortativity over time
        assort_df = pd.DataFrame(self.assort_records)
        assort_csv = os.path.join(self.output_dir, "assortativity.csv")
        assort_df.to_csv(assort_csv, index=False)

        # Network stats per turn
        net_df = pd.DataFrame(self.network_records)
        if self.save_run_network_stats:
            net_csv = os.path.join(self.output_dir, "network_stats.csv")
            net_df.to_csv(net_csv, index=False)
        else:
            net_csv = None  # will be provided in-memory to aggregator

        # No figures returned
        fig_path = None
        fig3_path = None
        fig2_path = None
        fig4_path = None

        return {
            "parameters_json": os.path.join(self.output_dir, "parameters.json"),
            "agents_csv": agents_csv,
            "agent_timeseries_csv": ts_csv,
            "correlations_csv": corr_csv,
            "assortativity_csv": assort_csv,
            "network_stats_csv": net_csv,  # may be None
            "network_stats_df": net_df if net_csv is None else None,  # in-memory copy if not saved
            "correlations_points_fig": fig_path,
            "assortativity_fig": fig4_path,
            "density_fig": fig2_path,
            "atlas_start_png": getattr(self, "atlas_start_path", None),
            "atlas_end_png": atlas_end_path,
        }


# ============
# Grid parsing
# ============

def parse_list(s: Any, cast):
    if s is None:
        return []
    if isinstance(s, (int, float)):
        return [cast(s)]
    if isinstance(s, str):
        toks = [t for t in re.split(r"[,\s]+", s.strip()) if t != ""]
        return [cast(t) for t in toks] if toks else []
    return [cast(s)]


def parse_payoff_sets(s: str):
    if s is None or str(s).strip() == "":
        return [{"R": 3.0, "P": 1.0, "T": 5.0, "S": 0.0}]
    sets = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [x.strip() for x in chunk.split(",")]
        if len(parts) != 4:
            raise ValueError(f"Invalid payoff quadruple '{chunk}'. Expected 'R,P,T,S'.")
        R, P, T, S = map(float, parts)
        if not (T > R > P > S):
            raise ValueError(
                f"Payoffs must satisfy T>R>P>S for a Prisoner's Dilemma. "
                f"Got T={T}, R={R}, P={P}, S={S}."
            )
        sets.append({"R": R, "P": P, "T": T, "S": S})
    return sets


def sanitize(s: Any) -> str:
    s = str(s).replace("/", "-").replace("\\", "-")
    return re.sub(r"[^A-Za-z0-9_.\-+]", "", s)


# ================
# Scenario helpers
# ================

def _scenario_beta_ab(scn: str) -> Dict[str, Tuple[float, float]]:
    """
    Return (a,b) per trait for scenario `scn`.
    Baseline => all Beta(5,5).
    *_hi  => Beta(7.5, 2.5) for that trait.
    *_lo  => Beta(2.5, 7.5) for that trait.
    """
    scn = scn.lower()
    base = {
        "agreeableness": (5.0, 5.0),
        "openness": (5.0, 5.0),
        "extraversion": (5.0, 5.0),
    }
    if scn in ("agreeableness_hi", "a_hi", "ahi"):
        base["agreeableness"] = (7.5, 2.5)
    elif scn in ("agreeableness_lo", "a_lo", "alo"):
        base["agreeableness"] = (2.5, 7.5)
    elif scn in ("openness_hi", "o_hi", "ohi"):
        base["openness"] = (7.5, 2.5)
    elif scn in ("openness_lo", "o_lo", "olo"):
        base["openness"] = (2.5, 7.5)
    elif scn in ("extraversion_hi", "e_hi", "ehi"):
        base["extraversion"] = (7.5, 2.5)
    elif scn in ("extraversion_lo", "e_lo", "elo"):
        base["extraversion"] = (2.5, 7.5)
    return base


def _scenario_short(scn: str) -> str:
    scn = scn.lower()
    mapping = {
        "baseline": "base",
        "agreeableness_hi": "Ahi",
        "a_hi": "Ahi",
        "ahi": "Ahi",
        "agreeableness_lo": "Alo",
        "a_lo": "Alo",
        "alo": "Alo",
        "openness_hi": "Ohi",
        "o_hi": "Ohi",
        "ohi": "Ohi",
        "openness_lo": "Olo",
        "o_lo": "Olo",
        "olo": "Olo",
        "extraversion_hi": "Ehi",
        "e_hi": "Ehi",
        "ehi": "Ehi",
        "extraversion_lo": "Elo",
        "e_lo": "Elo",
        "elo": "Elo",
    }
    return mapping.get(scn, "base")


# ================
# CI/aggregation utils (no plotting)
# ================

def mean_ci(series: Iterable[float]) -> Tuple[float, float, int, float]:
    vals = np.array([v for v in series if pd.notna(v)], dtype=float)
    n = len(vals)
    if n == 0:
        return (np.nan, np.nan, 0, np.nan)
    m = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    ci = 1.96 * sd / math.sqrt(n) if n > 1 else float("nan")
    return (m, sd, n, ci)


# ==============================
# Aggregate across seeds (CSV only)
# ==============================

def aggregate_across_seeds(root_outdir: str, summary_df: pd.DataFrame,
                           inmem_net_by_run: Optional[Dict[str, pd.DataFrame]] = None):
    inmem_net_by_run = inmem_net_by_run or {}

    group_cols = [
        "n_agents", "n_turns", "min_ideal_degree", "max_ideal_degree", "beta",
        "init_graph", "init_degree2_fraction", "init_edge_prob",
        "personality_source",
        "payoff_R", "payoff_P", "payoff_T", "payoff_S", "exchange_cost",
        "beta_extraversion_a", "beta_extraversion_b",
        "beta_openness_a", "beta_openness_b",
        "beta_agreeableness_a", "beta_agreeableness_b",
        "truncnorm_mean", "truncnorm_sd",
        "leniency_delta",
        "trait_scenario",
        "damping_lambda",
    ]
    for c in group_cols:
        if c not in summary_df.columns:
            summary_df[c] = np.nan

    def agg_tag(rowlike: dict) -> str:
        base = (
            f"N{sanitize(rowlike['n_agents'])}_T{sanitize(rowlike['n_turns'])}_"
            f"beta{sanitize(rowlike['beta'])}_mind{sanitize(rowlike['min_ideal_degree'])}_"
            f"maxd{sanitize(rowlike['max_ideal_degree'])}_"
            f"init{sanitize(rowlike['init_graph'])}_"
            f"R{sanitize(rowlike['payoff_R'])}_P{sanitize(rowlike['payoff_P'])}_"
            f"T{sanitize(rowlike['payoff_T'])}_S{sanitize(rowlike['payoff_S'])}_"
            f"C{sanitize(rowlike['exchange_cost'])}_psrc{sanitize(rowlike['personality_source'])}"
        )
        if str(rowlike["init_graph"]) == "bounded12":
            base += f"_deg2f{sanitize(rowlike['init_degree2_fraction'])}"
        else:
            base += f"_p{sanitize(rowlike['init_edge_prob'])}"
        base += f"_len{sanitize(rowlike['leniency_delta'])}"
        base += f"_lam{sanitize(rowlike['damping_lambda'])}"
        scn = str(rowlike.get("trait_scenario", "") or "")
        if scn:
            base += f"_scn{sanitize(_scenario_short(scn))}"
        return base

    def run_dir(row):
        return os.path.normpath(str(row["outdir"]))

    def ci95_from_sd_n(sd: pd.Series, n: pd.Series) -> pd.Series:
        df = (n - 1).clip(lower=1)
        crit = t.ppf(0.975, df)
        return (crit * sd / np.sqrt(n)).where(n > 1, np.nan)

    def gini_signed(v: np.ndarray, eps: float = 1e-10) -> float:
        """
        Normalised-by-absolute-mean Gini index:

            I = sum_{i,j} |x_i - x_j| /
                ( 2 (N-1) sum_i |x_i| )

        Handles negative values and collapses to 0 when all entries are zero.
        """
        x = np.asarray(v, dtype=float)
        x = x[np.isfinite(x)]
        n = x.size
        if n <= 1:
            return 0.0
        denom = np.sum(np.abs(x))
        if denom <= eps:
            return 0.0
        xs = np.sort(x)  # ascending
        i = np.arange(1, n + 1, dtype=float)
        # sum_{i,j} |x_j - x_i| == 2 * sum_i (2i - n - 1) * x_(i)
        diff_sum = 2.0 * np.sum((2.0 * i - n - 1.0) * xs)
        return float(diff_sum / (2.0 * (n - 1.0) * denom))

    agg_index_rows = []
    for key, g in summary_df.groupby(group_cols, dropna=False):
        key_dict = dict(zip(group_cols, key))
        expected_T = int(key_dict["n_turns"]) if not pd.isna(key_dict["n_turns"]) else None
        tag = agg_tag(key_dict)
        out_dir = os.path.join(root_outdir, "aggregate", tag)
        os.makedirs(out_dir, exist_ok=True)

        runs = []
        for _, row in g.iterrows():
            rdir = run_dir(row)
            runs.append(
                {
                    "seed": row.get("seed", np.nan),
                    "run_tag": str(row["run_tag"]),
                    "correlations": os.path.join(rdir, "correlations.csv"),
                    "assortativity": os.path.join(rdir, "assortativity.csv"),
                    "network_stats": os.path.join(rdir, "network_stats.csv"),
                    "agent_timeseries": os.path.join(rdir, "agent_timeseries.csv"),
                }
            )

        # --- Correlations (cumulative payoff vs metrics)
        corr_cat = []
        for r in runs:
            p = r["correlations"]
            if os.path.exists(p):
                df = pd.read_csv(p)
                df["seed"] = r["seed"]
                corr_cat.append(df)
        if corr_cat:
            corr_cat = pd.concat(corr_cat, ignore_index=True)
            agg_corr = (
                corr_cat.groupby(["turn", "variable"])["rho"]
                .agg(mean="mean", sd=lambda s: s.std(ddof=1), n="count")
                .reset_index()
            )
            agg_corr["ci95"] = ci95_from_sd_n(agg_corr["sd"], agg_corr["n"])
            agg_corr.to_csv(
                os.path.join(out_dir, "aggregated_correlations_points.csv"),
                index=False,
            )

        # --- Assortativity
        assort_cat = []
        for r in runs:
            p = r["assortativity"]
            if os.path.exists(p):
                df = pd.read_csv(p)
                df["seed"] = r["seed"]
                assort_cat.append(df)
        if assort_cat:
            assort_cat = pd.concat(assort_cat, ignore_index=True)
            agg_assort = (
                assort_cat.groupby(["turn", "variable"])["assortativity"]
                .agg(mean="mean", sd=lambda s: s.std(ddof=1), n="count")
                .reset_index()
            )
            agg_assort["ci95"] = ci95_from_sd_n(agg_assort["sd"], agg_assort["n"])
            agg_assort.to_csv(
                os.path.join(out_dir, "aggregated_assortativity.csv"), index=False
            )

        # --- Network stats
        net_cat = []
        for r in runs:
            p = r["network_stats"]
            if os.path.exists(p):
                df = pd.read_csv(p)
                df["seed"] = r["seed"]
                net_cat.append(df)
            else:
                mem = inmem_net_by_run.get(r["run_tag"])
                if mem is not None and len(mem):
                    df = mem.copy()
                    df["seed"] = r["seed"]
                    net_cat.append(df)
        if net_cat:
            net_cat = pd.concat(net_cat, ignore_index=True)
            agg_net = (
                net_cat.groupby("turn")
                .agg(
                    density_mean=("density", "mean"),
                    density_sd=("density", lambda s: s.std(ddof=1)),
                    density_n=("density", "count"),
                    avg_degree_mean=("avg_degree", "mean"),
                    avg_degree_sd=("avg_degree", lambda s: s.std(ddof=1)),
                    avg_degree_n=("avg_degree", "count"),
                    num_edges_mean=("num_edges", "mean"),
                    num_edges_sd=("num_edges", lambda s: s.std(ddof=1)),
                    num_edges_n=("num_edges", "count"),
                    avg_clustering_mean=("avg_clustering", "mean"),
                    avg_clustering_sd=(
                        "avg_clustering",
                        lambda s: s.std(ddof=1),
                    ),
                    avg_clustering_n=("avg_clustering", "count"),
                )
                .reset_index()
            )
            for base in ["density", "avg_degree", "num_edges", "avg_clustering"]:
                agg_net[f"{base}_ci95"] = ci95_from_sd_n(
                    agg_net[f"{base}_sd"], agg_net[f"{base}_n"]
                )
            agg_net.to_csv(
                os.path.join(out_dir, "aggregated_network_stats.csv"), index=False
            )

        # --- Panel side-bar metrics at final turn:
        #     mean points/agent, Gini (normalised by |mean|), harm prevalence
        panel_rows = []
        final_turns = []
        for r in runs:
            p = r["agent_timeseries"]
            if not os.path.exists(p):
                panel_rows.append(
                    {"seed": r["seed"], "mean_points": np.nan, "gini": np.nan, "harm": np.nan}
                )
                continue
            ats = pd.read_csv(p, usecols=["turn", "agent", "points"])
            if ats.empty:
                panel_rows.append(
                    {"seed": r["seed"], "mean_points": np.nan, "gini": np.nan, "harm": np.nan}
                )
                continue
            T_final = int(ats["turn"].max())
            if expected_T is not None:
                assert T_final == expected_T, \
                    f"Run {r['run_tag']} ended at turn {T_final}, expected {expected_T}."
            final_turns.append(T_final)
            v = ats.loc[ats["turn"] == T_final, "points"].to_numpy(dtype=float, copy=False)
            if v.size == 0:
                panel_rows.append(
                    {"seed": r["seed"], "mean_points": np.nan, "gini": np.nan, "harm": np.nan}
                )
                continue
            mean_pts = float(np.mean(v))
            gini = gini_signed(v)
            harm = float(np.mean(v < 0.0))
            panel_rows.append(
                {"seed": r["seed"], "mean_points": mean_pts, "gini": gini, "harm": harm}
            )

        if panel_rows:
            pr = pd.DataFrame(panel_rows)

            def summarize(col):
                vals = pr[col].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                n = len(vals)
                if n == 0:
                    return dict(mean=np.nan, sd=np.nan, n=0, ci95=np.nan)
                m = float(np.mean(vals))
                sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
                ci = float(t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else np.nan
                return dict(mean=m, sd=sd, n=n, ci95=ci)

            out = {"turn": int(max(final_turns)) if final_turns else np.nan}
            for name in ["mean_points", "gini", "harm"]:
                s = summarize(name)
                out[name] = s["mean"]
                out[f"{name}_sd"] = s["sd"]
                out[f"{name}_n"] = s["n"]
                out[f"{name}_ci95"] = s["ci95"]

            pd.DataFrame([out]).to_csv(
                os.path.join(out_dir, "aggregated_panel_metrics.csv"), index=False
            )

        agg_index_rows.append(
            {
                **key_dict,
                "aggregate_dir": out_dir,
                "n_seeds": g["seed"].nunique(),
                "tag": tag,
            }
        )

    agg_index = pd.DataFrame(agg_index_rows)
    if not agg_index.empty:
        path = os.path.join(root_outdir, "aggregate", "aggregate_index.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        agg_index.to_csv(path, index=False)


# ============
# Entry point
# ============

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Networked PD social simulation with personality-driven tie dynamics."
    )
    # All as strings to allow comma-separated lists
    p.add_argument("--n-agents", type=str)#, default="100")
    p.add_argument("--n-turns", type=str)#, default="300")
    p.add_argument("--seed", type=str)#, default="123")
    p.add_argument("--min-ideal-degree", type=str)#, default="1")
    p.add_argument("--max-ideal-degree", type=str)#, default="10")
    p.add_argument("--beta", type=str)#, default="0.5")
    p.add_argument("--init-graph", choices=["bounded12", "er"])#, default="bounded12")
    p.add_argument("--init-degree2-fraction", type=str)#, default="0.5")
    p.add_argument("--init-edge-prob", type=str)#, default="0.05")
    p.add_argument("--personality-source", type=str)#, default="beta")
    p.add_argument(
        "--payoff-sets",
        type=str,
        #default="3,1,5,0",
        help=(
            "Semicolon-separated list of payoff quadruples R,P,T,S "
            "(e.g., '3,1,5,0; 3,1,4.5,0'). Must satisfy T>R>P>S."
        ),
    )
    p.add_argument(
        "--exchange-cost",
        type=str,
        #default="2.0",
        help="Float or comma-separated list (grid across).",
    )
    p.add_argument(
        "--leniency-delta",
        type=str,
        #default="0.20",
        help="Leniency δ in [0,1): soft ideal = (1−δ)·D*. Default 0.20.",
    )
    p.add_argument(
        "--damping-lambda",
        type=str,
        #default="0.5",
        help="Global λ in [0,1], used in both tie addition and desertion.",
    )
    p.add_argument(
        "--trait-scenarios",
        type=str,
        #default=(
        #    "baseline,agreeableness_hi,agreeableness_lo,"
        #    "openness_hi,openness_lo,extraversion_hi,extraversion_lo"
        #),
        help=(
            "Comma-separated list of trait distribution scenarios. "
            "Use any of: baseline, agreeableness_hi, agreeableness_lo, "
            "openness_hi, openness_lo, extraversion_hi, extraversion_lo."
        ),
    )

    p.add_argument("--outdir", type=str, default=None)

    # Per-run output controls
    p.add_argument(
        "--per-run-plots",
        choices=["auto", "on", "off"],
        default="auto",
        help="Deprecated (no plots are produced).",
    )
    p.add_argument(
        "--per-run-network-stats",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "When 'auto', disable per-run network_stats.csv if multiple seeds "
            "are provided (aggregate only)."
        ),
    )

    # Beta params for traits (used if personality_source=beta and no scenario override)
    p.add_argument("--beta-extraversion-a", type=str, default="5.0")
    p.add_argument("--beta-extraversion-b", type=str, default="5.0")
    p.add_argument("--beta-openness-a", type=str, default="5.0")
    p.add_argument("--beta-openness-b", type=str, default="5.0")
    p.add_argument("--beta-agreeableness-a", type=str, default="5.0")
    p.add_argument("--beta-agreeableness-b", type=str, default="5.0")

    # Truncnorm params (used if personality_source=truncnorm)
    p.add_argument("--truncnorm-mean", type=str, default="0.5")
    p.add_argument("--truncnorm-sd", type=str, default="0.15")
    return p


def parse_args(argv=None) -> argparse.Namespace:
    p = build_arg_parser()
    if argv is None:
        args, _ = p.parse_known_args()  # Colab/Jupyter friendly
    else:
        args = p.parse_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)

    # Expand arguments into lists (for grid)
    n_agents_list = parse_list(args.n_agents, int) # or [100]
    n_turns_list = parse_list(args.n_turns, int) # or [300]
    seed_list = parse_list(args.seed, int) # or [123]
    min_deg_list = parse_list(args.min_ideal_degree, int) # or [1]
    max_deg_list = parse_list(args.max_ideal_degree, int) # or [10]
    beta_list = parse_list(args.beta, float) # or [0.5]
    init_graph_list = parse_list(args.init_graph, str) # or ["bounded12"]
    init_deg2_frac_list = parse_list(args.init_degree2_fraction, float) # or [0.5]
    init_edge_prob_list = parse_list(args.init_edge_prob, float) # or [0.05]
    personality_source_list = parse_list(args.personality_source, str) # or ["beta"]

    be_a_list = parse_list(getattr(args, "beta_extraversion_a"), float) # or [5.0]
    be_b_list = parse_list(getattr(args, "beta_extraversion_b"), float) # or [5.0]
    bo_a_list = parse_list(getattr(args, "beta_openness_a"), float) # or [5.0]
    bo_b_list = parse_list(getattr(args, "beta_openness_b"), float) # or [5.0]
    ba_a_list = parse_list(getattr(args, "beta_agreeableness_a"), float) # or [5.0]
    ba_b_list = parse_list(getattr(args, "beta_agreeableness_b"), float) # or [5.0]

    tn_mean_list = parse_list(getattr(args, "truncnorm_mean"), float) # or [0.5]
    tn_sd_list = parse_list(getattr(args, "truncnorm_sd"), float) # or [0.15]

    payoff_sets = parse_payoff_sets(args.payoff_sets)
    exchange_cost_list = parse_list(args.exchange_cost, float) # or [2.0]
    leniency_delta_list = parse_list(args.leniency_delta, float) # or [0.20]
    damping_lambda_list = parse_list(getattr(args, "damping_lambda"), float) # or [0.5]

    # Trait scenarios
    trait_scenario_list = [s.strip() for s in parse_list(args.trait_scenarios, str)] # or ["baseline"]]
    if any(s.lower() == "all" for s in trait_scenario_list):
        trait_scenario_list = [
            "baseline",
            "agreeableness_hi",
            "agreeableness_lo",
            "openness_hi",
            "openness_lo",
            "extraversion_hi",
            "extraversion_lo",
        ]

    # --- Sanity check: no grid dimension should be empty ---
    grid_dims = {
        "n_agents": n_agents_list,
        "n_turns": n_turns_list,
        "seed": seed_list,
        "min_ideal_degree": min_deg_list,
        "max_ideal_degree": max_deg_list,
        "beta": beta_list,
        "init_graph": init_graph_list,
        "init_degree2_fraction": init_deg2_frac_list,
        "init_edge_prob": init_edge_prob_list,
        "personality_source": personality_source_list,
        "payoff_sets": payoff_sets,
        "exchange_cost": exchange_cost_list,
        "leniency_delta": leniency_delta_list,
        "damping_lambda": damping_lambda_list,
        "trait_scenarios": trait_scenario_list,
    }

    for name, values in grid_dims.items():
        if not values:
            raise ValueError(
                f"No values provided for grid dimension '{name}'. "
                f"Did you forget to pass --{name.replace('_', '-')}?"
            )


    # Root output directory
    root_outdir = args.outdir or f"./grid_network_pd_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(root_outdir, exist_ok=True)

    # Determine multi-seed behavior (plots disabled anyway)
    multi_seed = len(seed_list) > 1

    def resolve_auto(flag: str, default_if_multi: bool, default_if_single: bool) -> bool:
        if flag == "on":
            return True
        if flag == "off":
            return False
        return default_if_single if not multi_seed else default_if_multi

    make_run_plots = False
    save_run_network_stats = resolve_auto(
        args.per_run_network_stats, default_if_multi=False, default_if_single=True
    )

    # Grid execution
    summary_rows = []
    inmem_net_by_run: Dict[str, pd.DataFrame] = {}
    combo_idx = 0

    for (
        n_agents,
        n_turns,
        seed,
        min_deg,
        max_deg,
        beta,
        init_graph,
        init_deg2_frac,
        init_edge_prob,
        psrc,
        be_a,
        be_b,
        bo_a,
        bo_b,
        ba_a,
        ba_b,
        tn_mean,
        tn_sd,
        payoff_set,
        exch_cost,
        len_delta,
        trait_scn,
        lam,
    ) in itertools.product(
        n_agents_list,
        n_turns_list,
        seed_list,
        min_deg_list,
        max_deg_list,
        beta_list,
        init_graph_list,
        init_deg2_frac_list,
        init_edge_prob_list,
        personality_source_list,
        be_a_list,
        be_b_list,
        bo_a_list,
        bo_b_list,
        ba_a_list,
        ba_b_list,
        tn_mean_list,
        tn_sd_list,
        payoff_sets,
        exchange_cost_list,
        leniency_delta_list,
        trait_scenario_list,
        damping_lambda_list,
    ):
        combo_idx += 1

        # Scenario-specific personality params (Beta) or truncnorm unchanged
        psrc_lower = psrc.lower()
        if psrc_lower == "beta":
            ab = _scenario_beta_ab(trait_scn)
            extraversion_a, extraversion_b = ab["extraversion"]
            openness_a, openness_b = ab["openness"]
            agreeableness_a, agreeableness_b = ab["agreeableness"]
            personality_params = {
                "extraversion": {
                    "a": float(extraversion_a),
                    "b": float(extraversion_b),
                },
                "openness": {
                    "a": float(openness_a),
                    "b": float(openness_b),
                },
                "agreeableness": {
                    "a": float(agreeableness_a),
                    "b": float(agreeableness_b),
                },
            }
        else:
            # For truncnorm, Beta params are metadata only (not used in sampling)
            extraversion_a, extraversion_b = be_a, be_b
            openness_a, openness_b = bo_a, bo_b
            agreeableness_a, agreeableness_b = ba_a, ba_b
            personality_params = {"mean": tn_mean, "sd": tn_sd}

        pay_tag = (
            f"R{sanitize(payoff_set['R'])}_P{sanitize(payoff_set['P'])}_"
            f"T{sanitize(payoff_set['T'])}_S{sanitize(payoff_set['S'])}_"
            f"C{sanitize(exch_cost)}"
        )
        tag = (
            f"N{sanitize(n_agents)}_T{sanitize(n_turns)}_seed{sanitize(seed)}_"
            f"beta{sanitize(beta)}_mind{sanitize(min_deg)}_maxd{sanitize(max_deg)}_"
            f"init{sanitize(init_graph)}_{pay_tag}"
        )
        if init_graph == "bounded12":
            tag += f"_deg2f{sanitize(init_deg2_frac)}"
        else:
            tag += f"_p{sanitize(init_edge_prob)}"
        tag += f"_psrc{sanitize(psrc)}"
        tag += f"_len{sanitize(len_delta)}"
        tag += f"_scn{_scenario_short(trait_scn)}"
        tag += f"_lam{sanitize(lam)}"

        outdir = os.path.join(root_outdir, tag)
        os.makedirs(outdir, exist_ok=True)

        sim = NetworkPDSim(
            n_agents=n_agents,
            n_turns=n_turns,
            seed=seed,
            min_ideal_degree=min_deg,
            max_ideal_degree=max_deg,
            beta=beta,
            init_graph=init_graph,
            init_degree2_fraction=init_deg2_frac,
            init_edge_prob=init_edge_prob,
            personality_source=psrc,
            personality_params=personality_params,
            payoff_R=payoff_set["R"],
            payoff_P=payoff_set["P"],
            payoff_T=payoff_set["T"],
            payoff_S=payoff_set["S"],
            exchange_cost=exch_cost,
            output_dir=outdir,
            make_run_plots=make_run_plots,
            save_run_network_stats=save_run_network_stats,
            leniency_delta=len_delta,
            trait_scenario=trait_scn,
            prior_strength_s=0.5,
            damping_lambda=lam,
        )
        sim.run()
        paths = sim.save_all()

        # If we suppressed per-run net stats, keep an in-memory copy for aggregation
        if not save_run_network_stats and isinstance(
            paths.get("network_stats_df"), pd.DataFrame
        ):
            inmem_net_by_run[tag] = paths["network_stats_df"]

        # Per-run summary row
        final_points_mean = float(np.mean(sim.points)) if sim.points.size else float("nan")
        try:
            if save_run_network_stats and paths["network_stats_csv"] is not None:
                net_df = pd.read_csv(paths["network_stats_csv"]).sort_values("turn")
            else:
                net_df = paths["network_stats_df"].sort_values("turn")
            final_density = float(net_df["density"].iloc[-1])
            final_avg_degree = float(net_df["avg_degree"].iloc[-1])
        except Exception:
            final_density = float("nan")
            final_avg_degree = float("nan")

        summary_rows.append(
            {
                "run_tag": tag,
                "outdir": outdir,
                "n_agents": n_agents,
                "n_turns": n_turns,
                "seed": seed,
                "min_ideal_degree": min_deg,
                "max_ideal_degree": max_deg,
                "beta": beta,
                "init_graph": init_graph,
                "init_degree2_fraction": init_deg2_frac,
                "init_edge_prob": init_edge_prob,
                "payoff_R": payoff_set["R"],
                "payoff_P": payoff_set["P"],
                "payoff_T": payoff_set["T"],
                "payoff_S": payoff_set["S"],
                "exchange_cost": exch_cost,
                "personality_source": psrc,
                "beta_extraversion_a": float(extraversion_a),
                "beta_extraversion_b": float(extraversion_b),
                "beta_openness_a": float(openness_a),
                "beta_openness_b": float(openness_b),
                "beta_agreeableness_a": float(agreeableness_a),
                "beta_agreeableness_b": float(agreeableness_b),
                "truncnorm_mean": tn_mean,
                "truncnorm_sd": tn_sd,
                "leniency_delta": float(len_delta),
                "damping_lambda": float(lam),
                "trait_scenario": trait_scn,
                "final_points_mean": final_points_mean,
                "final_density": final_density,
                "final_avg_degree": final_avg_degree,
                "parameters_json": paths["parameters_json"],
                "agents_csv": paths["agents_csv"],
                "agent_timeseries_csv": paths["agent_timeseries_csv"],
                "correlations_csv": paths["correlations_csv"],
                "assortativity_csv": paths["assortativity_csv"],
                "network_stats_csv": paths["network_stats_csv"],
                "correlations_points_fig": paths["correlations_points_fig"],
                "assortativity_fig": paths["assortativity_fig"],
                "density_fig": paths["density_fig"],
                "atlas_start_png": paths["atlas_start_png"],
                "atlas_end_png": paths["atlas_end_png"],
            }
        )
        print(f"[{combo_idx}] Completed run: {tag}")

    # Write grid summary at root
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(root_outdir, "grid_runs_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print("\n=== Grid complete ===")
    print("Root output directory:", root_outdir)
    print("Summary CSV:", summary_path)
    print(f"Total runs: {combo_idx}")
    print("Aggregating across seeds with 95% CIs ...")
    aggregate_across_seeds(root_outdir, summary_df, inmem_net_by_run)
    print("Aggregation complete. See:", os.path.join(root_outdir, "aggregate"))


if __name__ == "__main__":
    main()

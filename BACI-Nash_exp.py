# IMPORTS

import contextlib
import difflib
import faulthandler
import glob
import io
import itertools
import json
import math
import os
import re
import sys

import pandas as pd
import numpy as np
import networkx as nx
from scipy.linalg import eig


import gurobipy as gp
from gurobipy import GRB

GUROBI_ENV = gp.Env(params={"OutputFlag": 0})



# FILES AND CONSTANTS

FLOWS_GLOB = "BACI_HS92_Y*_V202601.csv"
COUNTRIES_FILE = "country_codes_V202601.csv"
SECTOR_DESCRIPTION_FILE = "product_codes_HS92_V202601.csv"

THETA_SAFETY = 1.5
EPS_SIGMA = 1.0
K_MARGIN = 1.5

DELTA_CDES = 0.5

# THETA CALIBRATION ON THE EQUILIBRIUM: theta is increased by this factor until every member
# has a strictly positive utility at the Nash equilibrium (the baseline positivity is already
# guaranteed by theta_0 = THETA_SAFETY * max_i theta_i^*)
THETA_EQ_STEP = 1.5
THETA_EQ_MAX_ITER = 60

# EXPERIMENT 3: deterministic perturbation of c0
EXP3_FRACTION = 0.10             # relative perturbation of the selected countries
EXP3_TOP_K = 5                   # case 1: number of top exporters among the members of S whose centrality is decreased
EXP3_ITALY_NAME = "italy"        # case 2: country whose centrality is increased
MEMBER_MARK = "\u2022"            # bullet marking the members of the coalition S in the new tables

LAMBDA_FRAC = 0.2
LB_TOL = 0.01                    # an arc is 'at the lower bound' if w_ij - ell_ij <= LB_TOL * ell_ij (solver tolerance)

# PRUNING LEVELS AND PERCENT FORMAT
PRUNING_FRACTIONS = (0.10,)
PERCENT_DECIMALS = 1

GAMMA_VALUE = 1.0
SEED = 20260101

MIQP_TIME_LIMIT = 1200
MIQP_GAP = 1e-6
MIQP_NUMERIC_FOCUS = 3

OUTSIDER_TOL = 1e-8
PROP3_TOL = 1e-6

EU_COUNTRY_CODES = frozenset({
    40,
    56,
    100,
    191,
    196,
    203,
    208,
    233,
    246,
    251,
    276,
    300,
    348,
    372,
    380,
    428,
    440,
    442,
    470,
    528,
    616,
    620,
    642,
    703,
    705,
    724,
    752,
})


MIDDLE_EAST_COUNTRY_CODES = frozenset({
    48, 275, 364, 368, 376, 400, 414, 422, 512, 634, 682, 760, 784, 887,
})


# DATA LOADING AND NETWORK CONSTRUCTION

def load_sector_descriptions(file_path):
    try:
        df = pd.read_csv(file_path, dtype={'code': str})
        df['code'] = df['code'].str.zfill(6)
        return dict(zip(df['code'], df['description']))
    except Exception as e:
        print(f"Warning: Could not load descriptions. Error: {e}")
        return {}


def load_flows_all_years(sector_codes, pattern=FLOWS_GLOB):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No BACI file found matching '{pattern}'.")

    frames = []
    years = []
    for path in files:
        match = re.search(r'_Y(\d{4})_', os.path.basename(path))
        if match is None:
            continue
        year = int(match.group(1))
        df = pd.read_csv(path, usecols=['i', 'j', 'k', 'v'], dtype={'k': str})
        df['k'] = df['k'].str.zfill(6)
        df = df[df['k'].isin(sector_codes)].copy()
        df['year'] = year
        frames.append(df)
        years.append(year)

    if len(years) < 2:
        raise ValueError(
            "At least two BACI years are needed to compute the historical "
            "volatility sigma_ij."
        )

    return pd.concat(frames, ignore_index=True), sorted(years)


def prune_flows(flows_df, fraction):
    a = fraction * flows_df['v'].sum() / len(flows_df)
    return flows_df[flows_df['v'] >= a].copy()


def sector_stats(flows_df, code, fraction):
    s_df = flows_df[flows_df['k'].astype(str).str.zfill(6) == str(code).zfill(6)].copy()
    if s_df.empty:
        return None, None, None, None, None, None
    arcs_before = len(s_df)
    flow_before = float(s_df['v'].sum())
    s_df = prune_flows(s_df, fraction)
    if s_df.empty:
        return None, None, None, None, None, None
    prune_info = {
        'arcs_before': arcs_before,
        'arcs_after': len(s_df),
        'flow_before': flow_before,
        'flow_after': float(s_df['v'].sum()),
        'mean_flow': flow_before / arcs_before,
        'threshold': fraction * flow_before / arcs_before,
    }
    G = nx.from_pandas_edgelist(s_df, source='i', target='j', edge_attr='v', create_using=nx.DiGraph())
    scc_list = list(nx.strongly_connected_components(G))
    if not scc_list:
        return None, None, None, None, None, None
    scc_nodes = max(scc_list, key=len)
    G_scc = G.subgraph(scc_nodes).copy()
    nodes = sorted(list(G_scc.nodes()))
    edges = list(G_scc.edges(data=True))
    return G_scc, nodes, edges, len(nodes), len(edges), prune_info


def compute_perron(G_scc, nodes):
    A0 = nx.to_numpy_array(G_scc, nodelist=nodes, weight='v')
    eigvals, eigvecs = eig(A0)
    idx = int(np.argmax(np.real(eigvals)))
    rho0 = float(np.real(eigvals[idx]))
    c0 = np.abs(np.real(eigvecs[:, idx]))
    c0 = c0 / np.sum(c0)
    return A0, rho0, c0


def build_out_in_structures(nodes, edges):
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)
    out_neighbors = {i: [] for i in range(n)}
    weight0 = {}
    in_from_all = {j: [] for j in range(n)}

    for (u, v, d) in edges:
        iu, iv = node_to_idx[u], node_to_idx[v]
        out_neighbors[iu].append(iv)
        weight0[(iu, iv)] = float(d['v'])
        in_from_all[iv].append((iu, float(d['v'])))

    return node_to_idx, out_neighbors, weight0, in_from_all


# COALITION SELECTION

def build_coalition_topk(export_vol_scc, node_to_idx, k):
    ranked_labels = list(export_vol_scc.head(k).index)
    S = sorted(node_to_idx[lbl] for lbl in ranked_labels)
    return S


class CoalitionNameError(Exception):
    pass


def _map_countries_scc(nodes, id_to_name):
    mapping = {}
    for idx, code in enumerate(nodes):
        key = id_to_name.get(code, str(code)).strip().lower()
        mapping.setdefault(key, []).append(idx)
    return mapping


def country_label(i, S, nodes, id_to_name):
    name = id_to_name.get(nodes[i], str(nodes[i]))
    return ("*" + name) if i in set(S) else name


def country_name(i, nodes, id_to_name):
    return id_to_name.get(nodes[i], str(nodes[i]))


def build_coalition_by_names(raw_names, nodes, id_to_name):
    mapping = _map_countries_scc(nodes, id_to_name)
    S = []
    for raw in raw_names:
        key = raw.strip().lower()
        if not key:
            continue
        matches = mapping.get(key)
        if not matches:
            suggestions = difflib.get_close_matches(key, mapping.keys(), n=3, cutoff=0.5)
            suggestion = (f" Did you mean: {', '.join(s.title() for s in suggestions)}?"
                          if suggestions else
                          " No suggestion found: check that the country is "
                          "present in the SCC for this sector (see the list above).")
            raise CoalitionNameError(f"'{raw}' not recognized.{suggestion}")
        if len(matches) > 1:
            codes = ', '.join(str(nodes[m]) for m in matches)
            raise CoalitionNameError(
                f"'{raw}' is ambiguous: {len(matches)} different countries in the SCC "
                f"share the same normalized name '{key}' (codes: {codes}). "
                f"Check country_codes_V202601.csv for a duplicated, empty, "
                f"or latin1->utf8 corrupted name, and use a more specific "
                f"name or the country code."
            )
        idx = matches[0]
        if idx in S:
            raise CoalitionNameError(
                f"'{raw}' ({country_name(idx, nodes, id_to_name)}) was "
                f"entered more than once."
            )
        S.append(idx)
    return sorted(S)


def build_coalition_eu(nodes, eu_codes=EU_COUNTRY_CODES):
    return sorted(idx for idx, code in enumerate(nodes) if code in eu_codes)


def build_coalition_mo(nodes, mo_codes=MIDDLE_EAST_COUNTRY_CODES):
    return sorted(idx for idx, code in enumerate(nodes) if code in mo_codes)


# FEASIBILITY DATA AND PARAMETER SPECIFICATION

def coalition_feasibility_data(S, out_neighbors, weight0, lambda_frac=LAMBDA_FRAC):
    T0, ell = {}, {}
    for i in S:
        T0[i] = sum(weight0[(i, j)] for j in out_neighbors[i])
        ell[i] = {j: lambda_frac * weight0[(i, j)] for j in out_neighbors[i]}
        T_tilde_i = T0[i] - sum(ell[i].values())
        if T_tilde_i < 0:
            raise ValueError(
                f"Member index {i}: lower bounds exceed the outgoing budget "
                f"(condition (12) violated, T_tilde = {T_tilde_i:.6g} < 0). "
                f"Decrease LAMBDA_FRAC."
            )
    return T0, ell, lambda_frac


def residual_budget(S, T0, ell):
    return {i: T0[i] - sum(ell[i].values()) for i in S}


def coalition_indegree(S, out_neighbors, n):
    d_S = {j: 0 for j in range(n)}
    for i in S:
        for j in out_neighbors[i]:
            d_S[j] += 1
    return d_S


def import_totals_scc(edges):
    totals = {}
    for _, destination, data in edges:
        totals[destination] = totals.get(destination, 0.0) + float(data['v'])
    return totals


def arc_volatility(hist_df, code, years, pairs):
    s_df = hist_df[hist_df['k'] == str(code).zfill(6)]
    table = s_df.groupby(['i', 'j', 'year'])['v'].sum().unstack('year', fill_value=0.0)
    table = table.reindex(columns=years, fill_value=0.0)
    sigma = table.std(axis=1, ddof=1)
    index = pd.MultiIndex.from_tuples(pairs, names=['i', 'j'])
    return sigma.reindex(index).fillna(0.0).to_numpy()


def build_alpha_j(S, out_neighbors, nodes, M_raw, id_to_name):
    destinations = sorted({j for i in S for j in out_neighbors[i]})
    bad = [j for j in destinations if M_raw.get(nodes[j], 0.0) <= 0.0]
    if bad:
        names = ', '.join(country_name(j, nodes, id_to_name) for j in bad)
        raise ValueError(
            f"Total imports M_j <= 0 for destination(s): {names}. "
            f"alpha_j = log(1 + M_j) would not be positive."
        )
    return {j: math.log1p(M_raw[nodes[j]]) for j in destinations}


def build_gamma_j_unit(S, out_neighbors, gamma_value=GAMMA_VALUE):
    destinations = {j for i in S for j in out_neighbors[i]}
    return {j: gamma_value for j in destinations}


# THETA (computed once per case from the baseline) AND COEFFICIENT CONFIGURATION

def compute_theta(S, out_neighbors, nodes, weight0, M_raw, kappa=THETA_SAFETY):
    """Baseline utility of member i with gamma_j = 1 and alpha_j = theta * L_j, L_j = ln(1 + M_j):
         u_i(w0) = theta * A_i - B_i,
         A_i = sum_j w0_ij L_j,   B_i = sum_j w0_ij (M_j - w0_ij).
       u_i(w0) > 0  <=>  theta > theta_i* = B_i / A_i.  theta = kappa * max_i theta_i*."""
    theta_i = {}
    for i in S:
        A_i = sum(weight0[(i, j)] * math.log1p(M_raw[nodes[j]]) for j in out_neighbors[i])
        B_i = sum(weight0[(i, j)] * (M_raw[nodes[j]] - weight0[(i, j)])
                  for j in out_neighbors[i])
        theta_i[i] = B_i / A_i
    binding = max(theta_i, key=theta_i.get)
    theta_min = theta_i[binding]
    theta = kappa * theta_min if theta_min > 0.0 else 1.0
    return theta, theta_min, binding, theta_i


def build_alpha_j_theta(S, out_neighbors, nodes, M_raw, id_to_name, theta):
    alpha_unit = build_alpha_j(S, out_neighbors, nodes, M_raw, id_to_name)
    return {j: theta * value for j, value in alpha_unit.items()}


def coefficients_experiment_1(S, out_neighbors, nodes, weight0, M_raw, id_to_name,
                              sigma_ij, d_S, theta):
    alpha_j = build_alpha_j_theta(S, out_neighbors, nodes, M_raw, id_to_name, theta)
    gamma_j = build_gamma_j_unit(S, out_neighbors)
    beta_ij, k_scale = build_beta_ij(S, out_neighbors, sigma_ij, d_S)
    label = ("Coefficients: alpha_j = theta * ln(1 + M_j), beta_ij = k/(sigma_ij + eps), gamma_j = 1")
    details = (f"theta = {theta:.6e} | k = {k_scale:.4f} | eps = {EPS_SIGMA} | gamma_j = {GAMMA_VALUE}")
    return alpha_j, beta_ij, gamma_j, label, details


EXPERIMENT_CONFIGURATIONS = [coefficients_experiment_1]


def build_beta_ij(S, out_neighbors, sigma_ij, d_S, eps=EPS_SIGMA, margin=K_MARGIN):
    arcs = [(i, j) for i in S for j in out_neighbors[i]]
    shared = [(sigma_ij[a] + eps) * (d_S[a[1]] - 1) for a in arcs if d_S[a[1]] > 1]
    if shared:
        threshold = max(shared)
    else:
        threshold = max(sigma_ij[a] + eps for a in arcs)
    k = margin * threshold
    beta_ij = {a: k / (sigma_ij[a] + eps) for a in arcs}
    return beta_ij, k


def check_diagonal_dominance(S, out_neighbors, beta_ij, d_S, gamma_j):
    violations = []
    for i in S:
        for j in out_neighbors[i]:
            if 2 * beta_ij[(i, j)] <= gamma_j[j] * (d_S[j] - 1):
                violations.append((i, j, d_S[j]))
    return len(violations) == 0, violations


def generate_random_c_des(c0, nodes, seed, delta=DELTA_CDES):
    factors = np.array([
        np.random.default_rng([int(seed), int(node)]).uniform(1.0 - delta, 1.0 + delta)
        for node in nodes
    ])
    c_des = np.asarray(c0, dtype=float) * factors
    return c_des / c_des.sum()


def top_exporters_among_members(S, export_vol_scc, node_to_idx, top_k=EXP3_TOP_K):
    """Node ids of the top_k exporters of the SCC among the members of the coalition S (by export volume)."""
    S_set = set(S)
    return [lbl for lbl in export_vol_scc.index if node_to_idx[lbl] in S_set][:top_k]


def generate_exp3_c_des(c0, nodes, id_to_name, export_vol_scc, node_to_idx, variant, S,
                        fraction=EXP3_FRACTION, top_k=EXP3_TOP_K):
    """Experiment 3 (deterministic target), one formula for both cases.
       P = set of perturbed countries, mu_i = multiplicative factor of country i in P:
           c_i^des = mu_i c_i^0   (i in P),      c_k^des = lam c_k^0   (k not in P),
           lam = (1 - sum_{i in P} mu_i c_i^0) / (1 - sum_{i in P} c_i^0).
       variant 1: P = top_k exporters among the members of the coalition S, mu_i = 1 - fraction.
       variant 2: P = {Italy},                    mu   = 1 + fraction.
       sum(c^des) = 1 and c^des > 0 if and only if mu_i > 0 and sum_{i in P} mu_i c_i^0 < 1."""
    c0 = np.asarray(c0, dtype=float)
    if variant == 1:
        perturbed = [node_to_idx[lbl] for lbl in top_exporters_among_members(S, export_vol_scc, node_to_idx, top_k)]
        mu = 1.0 - fraction
    else:
        italy = [idx for idx, code in enumerate(nodes)
                 if id_to_name.get(code, "").strip().lower() == EXP3_ITALY_NAME]
        if len(italy) != 1:
            raise ValueError("Italy is not present (or is ambiguous) in the SCC of this sector")
        perturbed = italy
        mu = 1.0 + fraction
    mass_P = float(np.sum(c0[perturbed]))
    new_mass_P = mu * mass_P
    if not (mu > 0 and new_mass_P < float(np.sum(c0))):
        raise ValueError(f"the perturbation is not admissible: it requires mu > 0 and "
                         f"sum_P mu c0 < sum c0 (here mu = {mu:g}, sum_P mu c0 = {new_mass_P:.6g}, "
                         f"sum c0 = {float(np.sum(c0)):.6g})")
    lam = (float(np.sum(c0)) - new_mass_P) / (float(np.sum(c0)) - mass_P)
    c_des = lam * c0
    c_des[perturbed] = mu * c0[perturbed]
    print(f"\n EXPERIMENT 3 - c^des: mu = {mu:g} on the perturbed countries, lambda = {lam:.9f} on the "
          f"other {len(c0) - len(perturbed)} countries (lambda = (1 - sum_P mu c0)/(1 - sum_P c0))")
    return c_des, perturbed


def print_top_tables(c0, S, nodes, id_to_name, export_vol_scc, node_to_idx):
    """Top 5 exporters among the members of S (with baseline centrality and its position in the ranking of
       the baseline centralities of the SCC) and top 5 baseline centralities. A bullet marks the members of S."""
    S_set = set(S)
    order = list(np.argsort(-np.asarray(c0)))
    position = {int(k): r + 1 for r, k in enumerate(order)}
    NAME_W = 34
    print(f"\n TOP {EXP3_TOP_K} EXPORTERS AMONG THE MEMBERS OF THE COALITION S, baseline centrality and its "
          f"position among the {len(nodes)} countries of the SCC ({MEMBER_MARK} = member of the coalition S)")
    print(" " + "-" * 100)
    print(f" {'Rank':>4}  {'Country':<{NAME_W}}{'Export volume':>18}{'c0':>18}{'Position of c0':>16}")
    for r, cid in enumerate(top_exporters_among_members(S, export_vol_scc, node_to_idx), start=1):
        vol = export_vol_scc[cid]
        k = node_to_idx[cid]
        mark = MEMBER_MARK if k in S_set else " "
        print(f" {r:>4}  {mark} {country_name(k, nodes, id_to_name):<{NAME_W - 2}}"
              f"{vol:>18.3f}{c0[k]:>18.6e}{position[k]:>16d}")
    print(" " + "-" * 100)
    print(f"\n TOP 5 HIGHEST BASELINE CENTRALITIES (SCC) ({MEMBER_MARK} = member of the coalition S)")
    print(" " + "-" * 60)
    print(f" {'Position':>8}  {'Country':<{NAME_W}}{'c0':>18}")
    for r, k in enumerate(order[:5], start=1):
        k = int(k)
        mark = MEMBER_MARK if k in S_set else " "
        print(f" {r:>8}  {mark} {country_name(k, nodes, id_to_name):<{NAME_W - 2}}{c0[k]:>18.6e}")
    print(" " + "-" * 60)


def print_exp3_table(c0, c_des, c_target, perturbed, S, nodes, id_to_name, variant):
    S_set = set(S)
    order = list(np.argsort(-np.asarray(c0)))
    position = {int(k): r + 1 for r, k in enumerate(order)}
    NAME_W = 34
    label = (f"TOP {EXP3_TOP_K} EXPORTERS AMONG THE MEMBERS OF S: c0 decreased" if variant == 1
             else "ITALY: c0 increased")
    print(f"\n EXPERIMENT 3 - PERTURBED COUNTRIES ({label} by {EXP3_FRACTION:.0%}; "
          f"{MEMBER_MARK} = member of the coalition S)")
    print(" " + "-" * 118)
    print(f" {'Country':<{NAME_W}}{'Position of c0':>15}{'c0':>16}{'c^des':>16}{'c^target':>16}"
          f"{'c^des/c0-1':>14}{'c^target/c0-1':>16}")
    for k in perturbed:
        mark = MEMBER_MARK if k in S_set else " "
        print(f" {mark} {country_name(k, nodes, id_to_name):<{NAME_W - 2}}{position[k]:>15d}"
              f"{c0[k]:>16.6e}{c_des[k]:>16.6e}{c_target[k]:>16.6e}"
              f"{c_des[k] / c0[k] - 1.0:>14.4%}{c_target[k] / c0[k] - 1.0:>16.4%}")
    print(" " + "-" * 118)
    n_neg = int(np.sum(np.asarray(c_des) < 0))
    print(f" Components of c^des: min = {float(np.min(c_des)):.6e} | negative components: {n_neg} "
          f"| sum = {float(np.sum(c_des)):.12f} (sum of c0 = {float(np.sum(c0)):.12f})")
    if n_neg:
        print(f" [WARNING] c^des has {n_neg} negative component(s): this should not happen with the "
              f"proportional redistribution (check the perturbation).")


def outsider_residual(c, S, A0, rho0):
    S_set = set(S)
    O = [k for k in range(len(c)) if k not in S_set]
    if not O:
        return 0.0
    return float(np.max(np.abs((A0[O, :] * c).sum(axis=1) - rho0 * c[O])) / (rho0 * np.max(c)))


def check_proposition_3(c, S, out_neighbors, ell, T_tilde, rho0, tol=PROP3_TOL):
    for i in S:
        lhs_fixed = sum(ell[i][j] * c[j] for j in out_neighbors[i])
        c_js = [c[j] for j in out_neighbors[i]]
        lower = T_tilde[i] * min(c_js)
        upper = T_tilde[i] * max(c_js)
        target = rho0 * c[i] - lhs_fixed
        if not (lower - tol <= target <= upper + tol):
            return False
    return True


def check_implementable_by_S(c, S, out_neighbors, ell, T_tilde, A0, rho0,
                              outsider_tol=OUTSIDER_TOL, prop3_tol=PROP3_TOL):
    prop3_ok = check_proposition_3(c, S, out_neighbors, ell, T_tilde, rho0, prop3_tol)
    outsider_res = outsider_residual(c, S, A0, rho0)
    outsiders_ok = outsider_res <= outsider_tol
    return prop3_ok and outsiders_ok, prop3_ok, outsiders_ok, outsider_res



# NUMERICAL DIAGNOSTICS (solver independent)

def print_centrality_statistics(c0, nodes, id_to_name):
    quantiles = np.quantile(c0, [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    print("\n BASELINE CENTRALITY c0: DYNAMIC RANGE")
    print("   quantiles (0, 1%, 5%, 25%, 50%, 75%, 95%, 100%): "
          + ", ".join(f"{q:.2e}" for q in quantiles))
    print(f"   nodes with c0 < 1e-6: {int(np.sum(c0 < 1e-6))} of {len(c0)} | "
          f"nodes with c0 < 1e-9: {int(np.sum(c0 < 1e-9))}")
    tiny = [idx for idx in np.argsort(c0) if c0[idx] < 1e-9][:15]
    if tiny:
        print("   smallest centralities: " + ", ".join(
            f"{id_to_name.get(nodes[idx], nodes[idx])} ({c0[idx]:.1e})" for idx in tiny))


def print_c0_table(c0, S, nodes, id_to_name):
    NAME_W = 34
    print("\n BASELINE CENTRALITY c0 (all countries of the SCC; * = member of the coalition)")
    print(f" {'Country':<{NAME_W}}{'c0':>18}")
    for k in range(len(nodes)):
        print(f" {country_label(k, S, nodes, id_to_name):<{NAME_W}}{c0[k]:>18.6e}")


def prop3_positions(c, S, out_neighbors, ell, T_tilde, rho0):
    positions = {}
    for i in S:
        neighbor_values = [c[j] for j in out_neighbors[i]]
        low, high = min(neighbor_values), max(neighbor_values)
        width = T_tilde[i] * (high - low)
        if len(out_neighbors[i]) < 2 or width <= 1e-30:
            positions[i] = (None, width)
            continue
        fixed = sum(ell[i][j] * c[j] for j in out_neighbors[i])
        positions[i] = ((rho0 * c[i] - fixed - T_tilde[i] * low) / width, width)
    return positions


def print_target_diagnostics(S, nodes, id_to_name, out_neighbors, ell, T_tilde, A0,
                             rho0, c0, c_target):
    outsiders = [k for k in range(len(c_target)) if k not in set(S)]
    if outsiders:
        residual = np.abs((A0[outsiders, :] * c_target).sum(axis=1) - rho0 * c_target[outsiders])
        absolute = float(np.max(residual))
        relative = absolute / (rho0 * float(np.max(c_target)))
    else:
        absolute = relative = 0.0
    print("\n TARGET DIAGNOSTICS")
    print(f"   outsider rows at the target: max |A c - rho c| = {absolute:.2e} "
          f"(relative to rho*max c: {relative:.2e}); sum of c = {float(np.sum(c_target)):.12f}")
    pos0 = prop3_positions(c0, S, out_neighbors, ell, T_tilde, rho0)
    pos1 = prop3_positions(c_target, S, out_neighbors, ell, T_tilde, rho0)
    NAME_W = 28
    print(f"   Position inside the Proposition 3 interval (0 = lower end, 1 = upper end;"
          f" * = interval width below 1e-6)")
    print(f"   {'Member':<{NAME_W}}{'arcs':>6}{'pos(c0)':>12}{'pos(target)':>14}")
    boundary0 = boundary1 = 0
    for i in S:
        p0, w0_width = pos0[i]
        p1, w1_width = pos1[i]
        text0 = "n.d." if p0 is None else f"{p0:.4f}" + ("*" if w0_width < 1e-6 else "")
        text1 = "n.d." if p1 is None else f"{p1:.4f}" + ("*" if w1_width < 1e-6 else "")
        if p0 is not None and (p0 <= 1e-6 or p0 >= 1 - 1e-6):
            boundary0 += 1
        if p1 is not None and (p1 <= 1e-6 or p1 >= 1 - 1e-6):
            boundary1 += 1
        print(f"   {country_label(i, S, nodes, id_to_name):<{NAME_W}}{len(out_neighbors[i]):>6}"
              f"{text0:>12}{text1:>14}")
    print(f"   members at an end of the interval (within 1e-6): c0 -> {boundary0}, "
          f"target -> {boundary1}")



# PROBLEM 2: TARGET-CENTRALITY MIQP (Gurobi)

def miqp_max_violation(c_vec, S, out_neighbors, ell, T_tilde, A0, rho0, eta):
    n = len(c_vec)
    outsiders = [k for k in range(n) if k not in set(S)]
    violations = [max(0.0, float(np.max(eta - c_vec))),
                  abs(float(np.sum(c_vec)) - 1.0)]
    for k in outsiders:
        violations.append(abs(float((A0[k, :] * c_vec).sum() - rho0 * c_vec[k])))
    for i in S:
        lhs_fixed_value = sum(ell[i][k] * c_vec[k] for k in out_neighbors[i])
        j_min = min(out_neighbors[i], key=lambda j: c_vec[j])
        j_max = max(out_neighbors[i], key=lambda j: c_vec[j])
        violations.append(max(0.0, lhs_fixed_value + T_tilde[i] * c_vec[j_min]
                              - rho0 * c_vec[i]))
        violations.append(max(0.0, rho0 * c_vec[i]
                              - (lhs_fixed_value + T_tilde[i] * c_vec[j_max])))
    return max(violations)


def solve_target_centrality_miqp(S, out_neighbors, T0, ell, T_tilde, A0, rho0,
                                  c_des, n, eta, c_init=None, verbose=False, tag=None):
    c_bar = 1.0 - (n - 1) * eta

    model = gp.Model("target_centrality_miqp", env=GUROBI_ENV)
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.TimeLimit = MIQP_TIME_LIMIT
    model.Params.MIPGap = MIQP_GAP
    model.Params.NumericFocus = MIQP_NUMERIC_FOCUS

    c = model.addVars(n, lb=eta, ub=1.0, name="c")

    outsiders = [k for k in range(n) if k not in S]
    for k in outsiders:
        model.addConstr(
            gp.quicksum(float(A0[k, j]) * c[j] for j in range(n)) == rho0 * c[k],
            name=f"outsider_eig_{k}"
        )

    model.addConstr(gp.quicksum(c[k] for k in range(n)) == 1.0, name="sum_to_one")

    multi = [i for i in S if len(out_neighbors[i]) >= 2]
    single = [i for i in S if len(out_neighbors[i]) == 1]

    z_min, z_max = {}, {}
    for i in multi:
        for j in out_neighbors[i]:
            z_min[i, j] = model.addVar(vtype=GRB.BINARY, name=f"zmin_{i}_{j}")
            z_max[i, j] = model.addVar(vtype=GRB.BINARY, name=f"zmax_{i}_{j}")
    model.update()

    for i in single:
        j = out_neighbors[i][0]
        model.addConstr(rho0 * c[i] == T0[i] * c[j], name=f"single_arc_{i}")

    print(f" MIQP: {len(multi)} member(s) with >= 2 arcs (big-M with binary selectors); "
          f"{len(single)} single-arc member(s) use the equality rho0*c_i = T0_i*c_j")

    for i in multi:
        lhs_fixed = gp.quicksum(ell[i][k] * c[k] for k in out_neighbors[i])
        M_i = max(0.0, c_bar * T0[i] - rho0 * eta)
        Mp_i = max(0.0, rho0 * c_bar - eta * T0[i])
        for j in out_neighbors[i]:
            model.addConstr(
                lhs_fixed + T_tilde[i] * c[j] <= rho0 * c[i] + M_i * (1 - z_min[i, j]),
                name=f"impl_min_{i}_{j}"
            )
            model.addConstr(
                rho0 * c[i] <= lhs_fixed + T_tilde[i] * c[j] + Mp_i * (1 - z_max[i, j]),
                name=f"impl_max_{i}_{j}"
            )
        model.addConstr(
            gp.quicksum(z_min[i, k] for k in out_neighbors[i]) == 1,
            name=f"zmin_sum_{i}"
        )
        model.addConstr(
            gp.quicksum(z_max[i, k] for k in out_neighbors[i]) == 1,
            name=f"zmax_sum_{i}"
        )

    obj = gp.quicksum((c[k] - float(c_des[k])) * (c[k] - float(c_des[k])) for k in range(n))
    model.setObjective(obj, GRB.MINIMIZE)

    if c_init is not None:
        c0_vec = np.asarray(c_init, dtype=float)
        for k in range(n):
            c[k].Start = float(c0_vec[k])
        for i in multi:
            nbrs = out_neighbors[i]
            j_min = min(nbrs, key=lambda j: c0_vec[j])
            j_max = max(nbrs, key=lambda j: c0_vec[j])
            for j in nbrs:
                z_min[i, j].Start = 1.0 if j == j_min else 0.0
                z_max[i, j].Start = 1.0 if j == j_max else 0.0
        start_violation = miqp_max_violation(c0_vec, S, out_neighbors, ell, T_tilde,
                                             A0, rho0, eta)
        print(f" MIQP warm start (c = c0) provided to Gurobi | max constraint violation "
              f"at the start = {start_violation:.2e} (feasibility tolerance 1e-6)")
        if start_violation > 1e-6:
            print("[WARNING] The warm start violates the constraints beyond the tolerance: "
                  "Gurobi will probably discard it.")

    model.optimize()

    if model.Status != GRB.OPTIMAL:
        if model.SolCount > 0:
            print(f"[WARNING] MIQP: Gurobi status {model.Status} (not proven optimal); "
                  f"using the best solution found.")
        else:
            raise RuntimeError(
                f"Gurobi did not find a solution for the target-centrality "
                f"MIQP (status {model.Status})."
            )

    c_star = np.array([c[k].X for k in range(n)])
    c_des_vec = np.asarray(c_des, dtype=float)
    distance_star = float(np.sum((c_star - c_des_vec) ** 2))
    print(f" MIQP diagnostics (Gurobi): status = {model.Status} | runtime = {model.Runtime:.1f}s | "
          f"relative gap = {model.MIPGap:.2e} | best bound = {model.ObjBound:.6g} | "
          f"nodes = {int(model.NodeCount)}")
    violation_star = miqp_max_violation(c_star, S, out_neighbors, ell, T_tilde,
                                        A0, rho0, eta)
    print(f"   max constraint violation at the returned solution = {violation_star:.2e}")
    if c_init is not None:
        c0_vec = np.asarray(c_init, dtype=float)
        print(f"   ||c0 - c_des||^2 = {float(np.sum((c0_vec - c_des_vec) ** 2)):.6g} (start) | "
              f"||c* - c_des||^2 = {distance_star:.6g} | "
              f"||c* - c0||^2 = {float(np.sum((c_star - c0_vec) ** 2)):.6g}")
    return c_star, distance_star


# PROBLEM 1: COALITION POTENTIAL GAME

def potential_gradient(i, S, out_neighbors, weight0, in_from_all, w, gamma_j, alpha_j, beta_ij):
    gradient = {}
    for j in out_neighbors[i]:
        others = sum(w[(k, j)] for k in S if k != i and j in out_neighbors[k])
        outsiders = sum(w0kj for (k, w0kj) in in_from_all[j] if k not in S)
        gradient[j] = (
            alpha_j[j]
            - 2.0 * beta_ij[(i, j)] * (w[(i, j)] - weight0[(i, j)])
            - gamma_j[j] * (others + outsiders)
        )
    return gradient


def align_dual_sign(S, out_neighbors, weight0, in_from_all, w_star, c, ell, T0,
                    gamma_j, alpha_j, beta_ij, mu_raw, pi_raw):
    residual = {1.0: 0.0, -1.0: 0.0}
    for i in S:
        gradient = potential_gradient(i, S, out_neighbors, weight0, in_from_all,
                                      w_star, gamma_j, alpha_j, beta_ij)
        free = [j for j in out_neighbors[i] if w_star[(i, j)] > ell[i][j] + 1e-6 * T0[i]]
        if not free:
            free = list(out_neighbors[i])
        for sign in (1.0, -1.0):
            residual[sign] += sum(
                abs(gradient[j] - sign * (mu_raw[i] + pi_raw[i] * c[j])) for j in free
            )
    return 1.0 if residual[1.0] <= residual[-1.0] else -1.0


# PROBLEM 1: COALITION POTENTIAL GAME


def print_infeasibility_diagnostic(model, S, out_neighbors, T0, ell, c, rho0, flow_scale,
                                   member_names):
    """Diagnostic for an infeasible Problem 1: (i) IIS computed by Gurobi; (ii) for every member,
       the exact distance of the target from the ends of its Proposition 3 interval,
         lo_i = sum_j ell_ij c_j + T~_i min c,   hi_i = sum_j ell_ij c_j + T~_i max c,
       compared with the Gurobi feasibility tolerance (1e-6) on the scaled centrality row
       (the row is divided by flow_scale)."""
    print("\n [DIAGNOSTIC] Infeasibility analysis of Problem 1")
    try:
        model.computeIIS()
        rows = [cn.ConstrName for cn in model.getConstrs() if cn.IISConstr]
        lbs = [v.VarName for v in model.getVars() if v.IISLB]
        print(f"   IIS: {len(rows)} constraint(s): {rows}")
        print(f"        {len(lbs)} variable lower bound(s) in the IIS"
              + (f": {lbs[:20]}" if lbs else ""))
    except Exception as exc:
        print(f"   IIS not available: {type(exc).__name__}: {exc}")
    print(f"   Feasibility margin of the centrality equality row (scaled by flow_scale = "
          f"{flow_scale:.3e}); Gurobi FeasibilityTol = 1e-6")
    print(f"   {'Member':<28}{'arcs':>6}{'rho c_i - lo_i':>18}{'hi_i - rho c_i':>18}"
          f"{'min margin/scale':>20}{'width/scale':>14}")
    worst = []
    for i in S:
        nb = out_neighbors[i]
        T_tilde = T0[i] - sum(ell[i][j] for j in nb)
        fixed = sum(ell[i][j] * c[j] for j in nb)
        lo = fixed + T_tilde * min(c[j] for j in nb)
        hi = fixed + T_tilde * max(c[j] for j in nb)
        target = rho0 * c[i]
        m_lo, m_hi = target - lo, hi - target
        margin = min(m_lo, m_hi) / flow_scale
        width = (hi - lo) / flow_scale
        worst.append((margin, i, len(nb), m_lo, m_hi, width))
    worst.sort(key=lambda t: t[0])
    for margin, i, na, m_lo, m_hi, width in worst:
        if margin > 1e-3 and len(worst) > 12:
            continue
        flag = "  <-- NEGATIVE (target outside the interval)" if margin < 0 else (
            "  <-- below tolerance" if margin < 1e-6 else "")
        print(f"   {member_names.get(i, str(i)):<28}{na:>6d}{m_lo:>18.6e}{m_hi:>18.6e}"
              f"{margin:>20.6e}{width:>14.3e}{flag}")
    n_neg = sum(1 for t in worst if t[0] < 0)
    n_tol = sum(1 for t in worst if 0 <= t[0] < 1e-6)
    print(f"   members with negative margin: {n_neg} | with 0 <= margin < 1e-6: {n_tol} "
          f"(of {len(worst)})")


def solve_coalition_game(S, out_neighbors, weight0, in_from_all, T0, ell, c,
                          rho0, d_S, gamma_j, alpha_j, beta_ij, verbose=False, tag=None,
                          member_names=None):
    dominant, violations = check_diagonal_dominance(S, out_neighbors, beta_ij, d_S, gamma_j)
    if not dominant:
        raise RuntimeError(
            f"Condition (Prop. 2) violated on {len(violations)} arc(s): the "
            f"potential P^S is not guaranteed strongly concave, so the problem "
            f"cannot be solved as a convex QP with the current parameters."
        )

    arcs = [(i, j) for i in S for j in out_neighbors[i]]
    m = len(arcs)
    index = {arc: a for a, arc in enumerate(arcs)}
    member_pos = {i: p for p, i in enumerate(S)}

    w0_vec = np.array([weight0[arc] for arc in arcs])
    beta_vec = np.array([beta_ij[arc] for arc in arcs])
    ell_vec = np.array([ell[i][j] for (i, j) in arcs])

    q = np.zeros(m)
    for a, (i, j) in enumerate(arcs):
        outsiders = sum(w0kj for (k, w0kj) in in_from_all[j] if k not in S)
        q[a] = alpha_j[j] - gamma_j[j] * outsiders + 2.0 * beta_vec[a] * w0_vec[a]

    H = np.diag(2.0 * beta_vec)
    for j in range(len(d_S)):
        members_into_j = [i for i in S if j in out_neighbors[i]]
        for i_, k_ in itertools.combinations(members_into_j, 2):
            a, b = index[(i_, j)], index[(k_, j)]
            H[a, b] += gamma_j[j]
            H[b, a] += gamma_j[j]

    A_budget = np.zeros((len(S), m))
    A_central = np.zeros((len(S), m))
    for a, (i, j) in enumerate(arcs):
        A_budget[member_pos[i], a] = 1.0
        A_central[member_pos[i], a] = c[j]
    b_budget = np.array([T0[i] for i in S])
    b_central = np.array([rho0 * c[i] for i in S])

    flow_scale = max(float(value) for value in T0.values())
    objective_scale = max(
        1.0,
        0.5 * flow_scale**2 * float(np.max(np.abs(H))),
        flow_scale * float(np.max(np.abs(q))),
    )


    model = gp.Model("coalition_potential_game", env=GUROBI_ENV)
    model.Params.OutputFlag = 1 if verbose else 0

    w_scaled = model.addVars(m, lb=(ell_vec / flow_scale).tolist(), name="w")
    model.update()

    budget_constr = {}
    central_constr = {}
    for i in S:
        p = member_pos[i]
        budget_idx = np.nonzero(A_budget[p])[0]
        central_idx = np.nonzero(A_central[p])[0]
        budget_constr[p] = model.addConstr(
            gp.quicksum(float(A_budget[p, a]) * w_scaled[int(a)] for a in budget_idx)
            == b_budget[p] / flow_scale,
            name=f"budget_{i}"
        )
        central_constr[p] = model.addConstr(
            gp.quicksum(float(A_central[p, a]) * w_scaled[int(a)] for a in central_idx)
            == b_central[p] / flow_scale,
            name=f"centrality_{i}"
        )

    objective = gp.QuadExpr()
    quad_scale = 0.5 * flow_scale**2 / objective_scale
    rows, cols = np.nonzero(H)
    for a, b_ in zip(rows.tolist(), cols.tolist()):
        objective.add(w_scaled[a] * w_scaled[b_], quad_scale * float(H[a, b_]))
    for a in range(m):
        objective.add(w_scaled[a], -flow_scale * float(q[a]) / objective_scale)

    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()

    used_fallback = False
    if model.Status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE):
        first_status = model.Status
        print(f"\n [FALLBACK] Problem 1 returned status {first_status}; "
              f"re-solving with Presolve=0 and NumericFocus=3")
        model.reset()
        model.Params.Presolve = 0
        model.Params.NumericFocus = 3
        model.optimize()
        used_fallback = True
        print(f" [FALLBACK] status after the re-solve = {model.Status}")

    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL) or (
            model.Status == GRB.SUBOPTIMAL and model.SolCount == 0):
        final_status = model.Status
        if final_status == GRB.INF_OR_UNBD:
            model.Params.DualReductions = 0
            model.optimize()
            final_status = model.Status
        names = {GRB.INFEASIBLE: "INFEASIBLE (3)", GRB.UNBOUNDED: "UNBOUNDED (5)"}
        print(f"\n [DIAGNOSTIC] Problem 1 not solved; status = "
              f"{names.get(final_status, final_status)}")
        if final_status == GRB.INFEASIBLE:
            print_infeasibility_diagnostic(model, S, out_neighbors, T0, ell, c, rho0,
                                           flow_scale, member_names or {})
        raise RuntimeError(f"Gurobi did not find an optimal solution (status {final_status}).")

    if model.Status == GRB.SUBOPTIMAL:
        print("[WARNING] Problem 1: Gurobi status 13 (suboptimal); "
              "using the returned solution.")

    w_vec = flow_scale * np.array([w_scaled[a].X for a in range(m)], dtype=float)
    budget_residual = float(np.max(np.abs((A_budget * w_vec).sum(axis=1) - b_budget)) / flow_scale)
    central_residual = float(np.max(np.abs((A_central * w_vec).sum(axis=1) - b_central)) / flow_scale)
    bound_violation = float(max(0.0, np.max(ell_vec - w_vec)) / flow_scale)
    print(f"\n GUROBI DIAGNOSTICS - coalition potential game (QP): status = {model.Status}"
          f"{' (after fallback: Presolve=0, NumericFocus=3)' if used_fallback else ''} | "
          f"runtime = {model.Runtime:.3f}s | barrier iterations = {int(model.BarIterCount)}")
    print(f"   residuals (relative to the largest budget): budget = {budget_residual:.2e} | "
          f"centrality = {central_residual:.2e} | lower bound = {bound_violation:.2e}")

    w_star = {arc: float(w_vec[a]) for a, arc in enumerate(arcs)}
    quadratic_value = math.fsum(
        float(w_vec[a]) * float(H[a, b]) * float(w_vec[b])
        for a in range(m)
        for b in range(m)
    )
    linear_value = math.fsum(float(q[a]) * float(w_vec[a]) for a in range(m))
    baseline_value = math.fsum(
        float(beta_vec[a]) * float(w0_vec[a]) ** 2 for a in range(m)
    )
    P_S_opt = float(-0.5 * quadratic_value + linear_value - baseline_value)

    dual_scale = objective_scale / flow_scale
    mu_raw = {
        i: dual_scale * float(budget_constr[member_pos[i]].Pi) for i in S
    }
    pi_raw = {
        i: dual_scale * float(central_constr[member_pos[i]].Pi) for i in S
    }
    sign = align_dual_sign(S, out_neighbors, weight0, in_from_all, w_star, c, ell, T0,
                           gamma_j, alpha_j, beta_ij, mu_raw, pi_raw)
    multipliers = {
        "pi": {i: sign * pi_raw[i] for i in S},
        "mu": {i: sign * mu_raw[i] for i in S},
    }

    return w_star, P_S_opt, multipliers


# K CHECK, SOCIAL WELFARE AND PRICE OF ANARCHY

def print_k_check(S, out_neighbors, sigma_ij, beta_ij, gamma_j, d_S):
    """Prints k (beta_ij = k/(sigma_ij + eps)), the bound k must exceed, and the conditions that
       depend on k: Prop. 2 (2 beta > gamma (d_S - 1)) and the sufficient condition for the strict
       concavity of W (beta > gamma (d_S - 1))."""
    arcs = [(i, j) for i in S for j in out_neighbors[i]]
    k_val = beta_ij[arcs[0]] * (sigma_ij[arcs[0]] + EPS_SIGMA)
    bound = k_val / K_MARGIN
    betas = np.array([beta_ij[a] for a in arcs])
    ratio = np.array([beta_ij[(i, j)] / (gamma_j[j] * (d_S[j] - 1))
                      for (i, j) in arcs if d_S[j] > 1])
    n_prop2 = int(np.sum(2.0 * ratio <= 1.0))
    n_conc = int(np.sum(ratio <= 1.0))
    print(f"\n CHECK k: k = {k_val:.6e} = {K_MARGIN:g} x bound | "
          f"bound = max_(i,j) (sigma_ij + eps)(d_S(j) - 1) = {bound:.6e} | "
          f"k > bound: {k_val > bound}")
    print(f"   eps = {EPS_SIGMA:g} | beta_ij = k/(sigma_ij + eps) in "
          f"[{betas.min():.4e}, {betas.max():.4e}]")
    if ratio.size:
        print(f"   min over arcs of beta_ij / (gamma_j (d_S(j)-1)) = {ratio.min():.4f} "
              f"(Prop. 2 needs > 0.5; concavity of W needs > 1)")
        print(f"   arcs violating Prop. 2: {n_prop2}/{ratio.size} | "
              f"arcs violating beta_ij > gamma_j (d_S(j)-1): {n_conc}/{ratio.size}")


def _is_positive_definite(block):
    """Pure-Python Cholesky (no LAPACK/BLAS call): True iff the symmetric matrix is PD."""
    n = len(block)
    L = [[0.0] * n for _ in range(n)]
    for r in range(n):
        for c in range(r + 1):
            acc = block[r][c] - math.fsum(L[r][t] * L[c][t] for t in range(c))
            if r == c:
                if acc <= 0.0:
                    return False
                L[r][r] = math.sqrt(acc)
            else:
                L[r][c] = acc / L[c][c]
    return True


def welfare_hessian_min_eig(S, out_neighbors, gamma_j, beta_ij, d_S):
    """Smallest eigenvalue of diag(beta) + G, block by block (one block per j). The Hessian of W is
       -2 (diag(beta) + G): W is strictly concave iff all the blocks are positive definite.
       Computed by bisection on lambda with a pure-Python Cholesky of block - lambda I
       (numpy.linalg.eigvalsh crashed natively in this environment)."""
    min_eig = math.inf
    for j in range(len(d_S)):
        members = [i for i in S if j in out_neighbors[i]]
        if not members:
            continue
        n = len(members)
        block = [[(gamma_j[j] if r != c else beta_ij[(members[r], j)]) for c in range(n)]
                 for r in range(n)]
        lo = min(beta_ij[(i, j)] for i in members) - gamma_j[j] * (n - 1) - 1.0
        hi = min(block[r][r] for r in range(n))
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            shifted = [[block[r][c] - (mid if r == c else 0.0) for c in range(n)]
                       for r in range(n)]
            if _is_positive_definite(shifted):
                lo = mid
            else:
                hi = mid
        min_eig = min(min_eig, lo)
    return min_eig


def social_welfare(w, S, out_neighbors, weight0, in_from_all, gamma_j, alpha_j, beta_ij):
    return math.fsum(member_utility(i, w, S, out_neighbors, weight0, in_from_all,
                                    gamma_j, alpha_j, beta_ij) for i in S)


def solve_social_optimum(S, out_neighbors, weight0, in_from_all, T0, ell, c, rho0, d_S,
                         gamma_j, alpha_j, beta_ij):
    """Maximizes W(w) = sum_{i in S} u_i(w) on K^S(c) (budget, centrality, w >= ell)."""
    arcs = [(i, j) for i in S for j in out_neighbors[i]]
    m = len(arcs)
    index = {arc: a for a, arc in enumerate(arcs)}
    member_pos = {i: p for p, i in enumerate(S)}
    w0_vec = np.array([weight0[arc] for arc in arcs])
    beta_vec = np.array([beta_ij[arc] for arc in arcs])
    ell_vec = np.array([ell[i][j] for (i, j) in arcs])

    q = np.zeros(m)
    for a, (i, j) in enumerate(arcs):
        outsiders = sum(w0kj for (k, w0kj) in in_from_all[j] if k not in S)
        q[a] = alpha_j[j] - gamma_j[j] * outsiders + 2.0 * beta_vec[a] * w0_vec[a]

    H = np.diag(beta_vec)
    for j in range(len(d_S)):
        members_into_j = [i for i in S if j in out_neighbors[i]]
        for i_, k_ in itertools.combinations(members_into_j, 2):
            a, b = index[(i_, j)], index[(k_, j)]
            H[a, b] += gamma_j[j]
            H[b, a] += gamma_j[j]

    flow_scale = max(float(value) for value in T0.values())
    objective_scale = max(1.0, flow_scale**2 * float(np.max(np.abs(H))),
                          flow_scale * float(np.max(np.abs(q))))
    model = gp.Model("social_optimum", env=GUROBI_ENV)
    model.Params.OutputFlag = 0
    w_scaled = model.addVars(m, lb=(ell_vec / flow_scale).tolist(), name="w")
    model.update()
    for i in S:
        model.addConstr(gp.quicksum(w_scaled[index[(i, j)]] for j in out_neighbors[i])
                        == T0[i] / flow_scale, name=f"so_budget_{i}")
        model.addConstr(gp.quicksum(float(c[j]) * w_scaled[index[(i, j)]]
                                    for j in out_neighbors[i])
                        == rho0 * c[i] / flow_scale, name=f"so_central_{i}")
    objective = gp.QuadExpr()
    rows, cols = np.nonzero(H)
    for a, b_ in zip(rows.tolist(), cols.tolist()):
        objective.add(w_scaled[a] * w_scaled[b_], flow_scale**2 * float(H[a, b_]) / objective_scale)
    for a in range(m):
        objective.add(w_scaled[a], -flow_scale * float(q[a]) / objective_scale)
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()
    if model.Status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE):
        model.reset()
        model.Params.Presolve = 0
        model.Params.NumericFocus = 3
        model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Social optimum: Gurobi status {model.Status}.")
    return {arc: flow_scale * w_scaled[a].X for a, arc in enumerate(arcs)}


def price_of_anarchy(S, out_neighbors, weight0, in_from_all, T0, ell, c, rho0, d_S,
                     gamma_j, alpha_j, beta_ij, w_ne):
    """Returns a dict with W(w^NE), W(w^SO), PoA = W(w^SO)/W(w^NE) (None if not computable)."""
    out = {'W_NE': social_welfare(w_ne, S, out_neighbors, weight0, in_from_all,
                                  gamma_j, alpha_j, beta_ij),
           'W_SO': None, 'poa': None, 'min_eig': None, 'note': ''}
    out['min_eig'] = welfare_hessian_min_eig(S, out_neighbors, gamma_j, beta_ij, d_S)
    if out['min_eig'] <= 1e-12 * max(beta_ij.values()):
        out['note'] = "W not strictly concave: social optimum not unique/not computed"
        return out
    w_so = solve_social_optimum(S, out_neighbors, weight0, in_from_all, T0, ell, c, rho0,
                                d_S, gamma_j, alpha_j, beta_ij)
    out['W_SO'] = social_welfare(w_so, S, out_neighbors, weight0, in_from_all,
                                 gamma_j, alpha_j, beta_ij)
    if out['W_NE'] > 0.0:
        out['poa'] = out['W_SO'] / out['W_NE']
    else:
        out['note'] = "W(w^NE) <= 0: ratio not meaningful"
    return out


def compute_lower_bound_multipliers(S, out_neighbors, weight0, in_from_all, w_star, c,
                                    gamma_j, alpha_j, beta_ij, multipliers):
    """Multipliers nu_ij >= 0 of the lower bounds w_ij >= ell_ij (paper, Lagrangian (21)).
       From the stationarity condition (22):
         nu_ij = pi_i c_j + mu_i - [alpha_ij - 2 beta_ij (w_ij - w0_ij) - gamma_j (sum_k w_kj + sum_out w0_kj)].
       By complementary slackness nu_ij = 0 on the arcs with w_ij > ell_ij."""
    nu = {}
    for i in S:
        grad = potential_gradient(i, S, out_neighbors, weight0, in_from_all, w_star,
                                  gamma_j, alpha_j, beta_ij)
        for j in out_neighbors[i]:
            nu[(i, j)] = multipliers["pi"][i] * c[j] + multipliers["mu"][i] - grad[j]
    return nu


KKT_FEAS_TOL = 1e-6   # primal feasibility (budget, centrality, lower bounds), relative to the largest budget (Gurobi FeasibilityTol)
KKT_SIGN_TOL = 1e-6   # nu_ij >= 0 holds up to KKT_SIGN_TOL * max(1, max_j |nu_ij|) (member i)
KKT_COMP_TOL = 1e-4   # complementarity nu_ij (w_ij - ell_ij) = 0 holds up to KKT_COMP_TOL * |P^S| (barrier accuracy)
LICQ_C_TOL = 1e-9     # two centralities c_r, c_s are 'different' if |c_r - c_s| > LICQ_C_TOL * max(c_r, c_s)


def verify_multipliers(S, out_neighbors, w_star, c, ell, T0, rho0, nu, multipliers, P_S_opt):
    """Verification of the multipliers of each member i in S (no effect on the results).
       Prop. 4 (certificate of the Gurobi solution): the KKT system (22)-(27) holds, i.e. stationarity (22)
         [satisfied by construction, nu_ij is computed from it], primal feasibility (budget, centrality,
         lower bounds), nu_ij >= 0 (25) and complementarity nu_ij (w_ij - ell_ij) = 0 (26)-(27).
       Prop. 5 (uniqueness of the multipliers): LICQ for member i, i.e. the free-arc set F_i = {j : w_ij > ell_ij}
         contains two arcs r, s with c_r != c_s (equivalently rank [c_Fi 1_Fi] = 2).
       Returns {i: {'valid': bool, 'licq': bool, 'unique': bool}}, unique = valid AND licq."""
    flow_scale = max(float(v) for v in T0.values())
    comp_scale = max(1.0, abs(float(P_S_opt)))
    out = {}
    for i in S:
        arcs_i = list(out_neighbors[i])
        w = {j: w_star[(i, j)] for j in arcs_i}
        l = {j: ell[i][j] for j in arcs_i}
        # Prop. 4
        ok = abs(sum(w.values()) - T0[i]) <= KKT_FEAS_TOL * flow_scale
        ok = ok and abs(sum(c[j] * w[j] for j in arcs_i) - rho0 * c[i]) <= KKT_FEAS_TOL * flow_scale
        ok = ok and all(w[j] >= l[j] - KKT_FEAS_TOL * flow_scale for j in arcs_i)
        nu_scale = max(1.0, max(abs(nu[(i, j)]) for j in arcs_i))
        ok = ok and all(nu[(i, j)] >= -KKT_SIGN_TOL * nu_scale for j in arcs_i)
        ok = ok and all(abs(nu[(i, j)] * (w[j] - l[j])) <= KKT_COMP_TOL * comp_scale for j in arcs_i)
        # Prop. 5
        free = [j for j in arcs_i if w[j] - l[j] > LB_TOL * l[j]]
        licq = False
        if len(free) >= 2:
            cf = [float(c[j]) for j in free]
            licq = (max(cf) - min(cf)) > LICQ_C_TOL * max(cf)
        out[i] = {'valid': bool(ok), 'licq': bool(licq), 'unique': bool(ok and licq)}
    return out


def print_lower_bound_multipliers(S, id_to_name, nodes, out_neighbors, nu, w_star, ell, T0):
    at_bound = {a: (w_star[a] - ell[a[0]][a[1]]) <= LB_TOL * ell[a[0]][a[1]] for a in nu}
    scale = max(1.0, max(abs(v) for v in nu.values()))
    free_vals = [abs(nu[a]) for a in nu if not at_bound[a]]
    bound_vals = [nu[a] for a in nu if at_bound[a]]
    NAME_W = 32
    NUM_W = 22
    ROW_W = NAME_W * 2 + NUM_W * 3 + 4
    print("\n" + "=" * ROW_W)
    print("MOLTIPLICATORI nu_ij DEI LOWER BOUND w_ij >= ell_ij (paper, eq. (21)-(22))")
    print("=" * ROW_W)
    print(f" arcs at the lower bound: {len(bound_vals)}/{len(nu)} | listed below; "
          f"nu_ij should be 0 on the other arcs (max |nu_ij| there = "
          f"{(max(free_vals) if free_vals else 0.0):.3e}, relative to max |nu| {scale:.3e}: "
          f"{(max(free_vals) if free_vals else 0.0) / scale:.2e})")
    if bound_vals:
        print(f" min nu_ij over the arcs at the bound = {min(bound_vals):.6e} (must be >= 0)")
    print(f" nu_ij computed from the stationarity condition; only the ACTIVE constraints (w_ij at the lower bound) "
          f"are listed, ordered by exporter i (members of S) and then by importer j")
    print(f" {'Paese i':<{NAME_W}} {'Paese j':<{NAME_W}}{'w_ij^NE':>{NUM_W}}"
          f"{'ell_ij':>{NUM_W}}{'nu_ij':>{NUM_W}}")
    print("-" * ROW_W)
    print(f" ACTIVE constraints (w_ij at the lower bound): {len(bound_vals)} arcs")
    print("-" * ROW_W)
    order_S = {i: p for p, i in enumerate(S)}
    arcs_block = [a for a in nu if at_bound[a]]
    for (i, j) in sorted(arcs_block, key=lambda a: (order_S[a[0]], country_name(a[1], nodes, id_to_name))):
        print(f" {country_label(i, S, nodes, id_to_name):<{NAME_W}} "
              f"{country_label(j, S, nodes, id_to_name):<{NAME_W}}"
              f"{w_star[(i, j)]:>{NUM_W}.3f}{ell[i][j]:>{NUM_W}.3f}{nu[(i, j)]:>{NUM_W}.6e}")
    print("=" * ROW_W)
    return len(bound_vals)


def member_utility(i, w, S, out_neighbors, weight0, in_from_all, gamma_j, alpha_j, beta_ij):
    u = 0.0
    for j in out_neighbors[i]:
        wij = w[(i, j)]
        w0ij = weight0[(i, j)]
        outsiders = sum(w0kj for (k, w0kj) in in_from_all[j] if k not in S)
        members_others = sum(w[(k, j)] for k in S if k != i and j in out_neighbors[k])
        u += alpha_j[j] * wij - beta_ij[(i, j)] * (wij - w0ij) ** 2 - gamma_j[j] * wij * (members_others + outsiders)
    return u


# OUTPUT

def print_c_vector(c_vec, nodes, id_to_name, label):
    NAME_W = 34
    NUM_W = 16
    ROW_W = NAME_W + NUM_W

    print("\n" + "-" * ROW_W)
    print(f" {label}")
    print("-" * ROW_W)
    print(f" {'Country':<{NAME_W}}{'c_k':>{NUM_W}}")
    for idx in range(len(nodes)):
        name = country_name(idx, nodes, id_to_name)
        print(f" {name:<{NAME_W}}{c_vec[idx]:>{NUM_W}.6f}")
    print("-" * ROW_W)


def print_results(S, id_to_name, out_neighbors, weight0, in_from_all, nodes,
                   w_star, P_S_opt, gamma_j, alpha_j, beta_ij, dominant, multipliers):
    NAME_W = 32
    NUM_W = 20
    PCT_W = 14
    ROW_W = NAME_W * 2 + NUM_W * 2 + PCT_W + 5

    def percent_change(new_value, baseline_value):
        if baseline_value == 0:
            return "n.d."
        return f"{(new_value - baseline_value) / abs(baseline_value) * 100:+.2f}%"

    print("\n" + "=" * ROW_W)
    print("RISULTATI DEL GIOCO DI COALIZIONE")
    print("=" * ROW_W)
    print(f" |S| = {len(S)} member(s); uniqueness (Prop. 2): "
          f"{'YES' if dominant else 'NOT GUARANTEED'}")
    print(f" P^S(w_S^NE; c) = {P_S_opt:.6f}")

    u_baseline = {i: member_utility(i, weight0, S, out_neighbors, weight0,
                                     in_from_all, gamma_j, alpha_j, beta_ij)
                  for i in S}
    u_ne = {i: member_utility(i, w_star, S, out_neighbors, weight0,
                               in_from_all, gamma_j, alpha_j, beta_ij)
            for i in S}

    print("\n" + "=" * ROW_W)
    print("FLUSSI DEI MEMBRI DELLA COALIZIONE")
    print("=" * ROW_W)
    header = (f" {'Paese i':<{NAME_W}} {'Paese j':<{NAME_W}}"
              f" {'w_ij^0 (baseline)':>{NUM_W}}"
              f" {'w_ij^NE (ottimo)':>{NUM_W}}"
              f" {'Delta_i (%)':>{PCT_W}}")
    print(header)
    for i in S:
        name_i = country_label(i, S, nodes, id_to_name)
        for j in out_neighbors[i]:
            name_j = country_label(j, S, nodes, id_to_name)
            w0ij = weight0[(i, j)]
            wij = w_star[i, j]
            rel_delta = (wij - w0ij) / w0ij * 100.0
            print(f" {name_i:<{NAME_W}} {name_j:<{NAME_W}}"
                  f" {w0ij:>{NUM_W}.3f} {wij:>{NUM_W}.3f}"
                  f" {rel_delta:>+{PCT_W - 1}.2f}%")

    print("\n" + "=" * ROW_W)
    print("UTILITA DEI MEMBRI DELLA COALIZIONE")
    print("=" * ROW_W)
    utility_width = 24
    header2 = (f" {'Paese i':<{NAME_W}}"
               f" {'u_i(w^0)':>{utility_width}}"
               f" {'u_i(w^NE)':>{utility_width}}"
               f" {'Delta_u_i (%)':>{PCT_W}}")
    print(header2)
    for i in S:
        name_i = country_label(i, S, nodes, id_to_name)
        u0, uNE = u_baseline[i], u_ne[i]
        delta_u = percent_change(uNE, u0)
        print(f" {name_i:<{NAME_W}} {u0:>{utility_width}.6f}"
              f" {uNE:>{utility_width}.6f} {delta_u:>{PCT_W}}")

    print("\n" + "=" * ROW_W)
    print("MOLTIPLICATORI DI LAGRANGE ALL'EQUILIBRIO")
    print("=" * ROW_W)
    header3 = (f" {'Paese i':<{NAME_W}}"
               f" {'pi_i (centralita)':>{utility_width}}"
               f" {'mu_i (budget)':>{utility_width}}")
    print(header3)
    for i in S:
        name_i = country_label(i, S, nodes, id_to_name)
        pi_i = multipliers["pi"][i]
        mu_i = multipliers["mu"][i]
        print(f" {name_i:<{NAME_W}} {pi_i:>{utility_width}.6f}"
              f" {mu_i:>{utility_width}.6f}")
    print("=" * ROW_W)


# MAIN

class _TeeStream:
    def __init__(self, console, log_file):
        self._console = console
        self._log_file = log_file

    def write(self, text):
        console_result = self._console.write(text)
        self._log_file.write(text)
        return console_result

    def flush(self):
        self._console.flush()
        self._log_file.flush()

    def isatty(self):
        return self._console.isatty()

    def fileno(self):
        return self._console.fileno()

    @property
    def encoding(self):
        return self._console.encoding

    @property
    def errors(self):
        return self._console.errors


def configure_output_log():
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_output.txt")
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    original_streams = (sys.stdout, sys.stderr)
    sys.stdout = _TeeStream(sys.stdout, log_file)
    sys.stderr = _TeeStream(sys.stderr, log_file)
    faulthandler.enable(file=log_file)
    print(f"Full run output is being saved to: {log_path}", flush=True)
    return log_file, original_streams


def count_signs(values):
    positive = sum(1 for v in values if v > 0)
    negative = sum(1 for v in values if v < 0)
    return positive, negative


def pct(value, decimals=PERCENT_DECIMALS):
    return f"{100.0 * value:.{decimals}f}%"


def pct_frac(value):
    return f"{100.0 * value:.2f}".rstrip('0').rstrip('.') + '%'


def print_experiment_summary(code, desc, rows, experiment=None):
    print("\n" + "=" * 173)
    print(f"EXPERIMENT {experiment} SUMMARY - SECTOR {code}: {desc}")
    print("=" * 173)
    header = (f" {'Pruning':>8}{'theta':>13}{'Threshold a':>14}{'Arcs kept':>24}"
              f"{'Flow kept':>11}{'P^S':>14}{'u>0 (base)':>12}{'u<0 (base)':>12}"
              f"{'u>0 (NE)':>10}{'u<0 (NE)':>10}{'||c*-cdes||':>13}{'PoA':>10}")
    print(header)
    for row in rows:
        head = f" {pct_frac(row['fraction']):>8}"
        if row['error'] is not None:
            print(f"{head}  not available: {row['error']}")
            continue
        poa_txt = f"{row['poa']:.4f}" if row['poa'] is not None else 'n.d.'
        arcs_txt = f"{row['arcs_after']}/{row['arcs_before']} = {pct(row['arcs_ratio'])}"
        print(f"{head}{row['theta']:>13.3e}{row['threshold']:>14.3e}{arcs_txt:>24}"
              f"{pct(row['flow_ratio']):>11}"
              f"{row['potential']:>14.3e}{row['base_pos']:>12d}{row['base_neg']:>12d}"
              f"{row['ne_pos']:>10d}{row['ne_neg']:>10d}"
              f"{row['dist_target']:>13.3e}"
              f"{poa_txt:>10}")
    print("=" * 173)


def print_final_vectors(cases):
    kind_text = {'c0': "c^target = c0 (experiment 1)",
                 'cdes': "c^target = c^des (desired target, implementable: MIQP not needed)",
                 'cstar': "c^target = c* (MIQP solution)"}
    kind_head = {'c0': 'c^target=c0', 'cdes': 'c^target=c^des', 'cstar': 'c^target=c*'}
    for case in cases:
        nodes, id_to_name, S = case['nodes'], case['id_to_name'], set(case['S'])
        c0, c_target, kind = case['c0'], case['c_target'], case['target_kind']
        NAME_W = 34
        show_des = case['experiment'] in (2, 3)
        c_des = case['c_des']
        width = 100 + (18 if show_des else 0)
        extra = (f" | CASE {case['exp3_case']}" if case['experiment'] == 3 else "")
        print("\n" + "=" * width)
        print(f"CENTRALITY VECTORS - SECTOR {case['code']} | PRUNING {pct_frac(case['fraction'])} "
              f"| EXPERIMENT {case['experiment']}{extra} (* = member of the coalition)")
        print(f" {kind_text[kind]}")
        print("=" * width)
        print(f" {'Country':<{NAME_W}}{'c0':>18}" + (f"{'c^des':>18}" if show_des else "")
              + f"{kind_head[kind]:>18}")
        for k in range(len(nodes)):
            mark = "*" if k in S else " "
            print(f" {mark}{country_name(k, nodes, id_to_name):<{NAME_W - 1}}"
                  f"{c0[k]:>18.6e}" + (f"{c_des[k]:>18.6e}" if show_des else "")
                  + f"{c_target[k]:>18.6e}")
        print("=" * width)


# COALITION CHOICE (asked once per sector, resolved in the network of every pruning level)

def resolve_coalition(raw_choice, top_k, nodes, node_to_idx, export_vol_scc, id_to_name):
    choice = raw_choice.strip().lower()
    if choice in ("", "t", "top", "top-k"):
        S = build_coalition_topk(export_vol_scc, node_to_idx, min(top_k, len(nodes)))
        return S, f"automatic top-{len(S)}"
    if choice in ("eu", "eu27", "european union"):
        S = build_coalition_eu(nodes)
        if not S:
            raise ValueError("no EU member is present in this SCC")
        return S, f"{len(S)}/{len(EU_COUNTRY_CODES)} EU members present in this SCC"
    if choice in ("mo", "middle east"):
        S = build_coalition_mo(nodes)
        if not S:
            raise ValueError("no Middle East member is present in this SCC")
        return S, f"{len(S)}/{len(MIDDLE_EAST_COUNTRY_CODES)} Middle East members present in this SCC"
    S = build_coalition_by_names(raw_choice.split(','), nodes, id_to_name)
    return S, "manual selection"


def ask_coalition(code, nodes, node_to_idx, export_vol_scc, id_to_name):
    raw = input(
        f"\n Coalition for sector {code} (used for every pruning level): press ENTER or type "
        "'top' for automatic top-k, 'eu' for the European Union, 'mo' for the Middle East, "
        "or list the countries separated by commas (country names as in the c0 table / TOP 5 EXPORTERS list of the sector output): "
    ).strip()
    while True:
        top_k = None
        try:
            if raw.lower() in ("", "t", "top", "top-k"):
                top_k = int(input(f" Enter coalition size k (1 <= k <= {len(nodes)}): "))
            resolve_coalition(raw, top_k, nodes, node_to_idx, export_vol_scc, id_to_name)
            return raw, top_k
        except (CoalitionNameError, ValueError) as exc:
            print(f"  [ERROR] {exc}")
            raw = input(" Enter 'top', 'eu', 'mo' or the countries separated by commas: ").strip()


# CASE PREPARATION: NETWORK, COALITION, TARGET (PROBLEM 2). One case = (sector, pruning level)

def prepare_case(code, desc, fraction, flows_df, hist_df, years, baseline_year,
                 id_to_name, coalition_state, experiment=2, exp3_case=None):
    G_scc, nodes, edges, n_n, n_e, prune_info = sector_stats(flows_df, code, fraction)
    if G_scc is None:
        print(f"\n[INFO] Sector {code}, pruning {pct_frac(fraction)}: no network available "
              f"in baseline year {baseline_year}.")
        return None

    A0, rho0, c0 = compute_perron(G_scc, nodes)
    node_to_idx, out_neighbors, weight0, in_from_all = build_out_in_structures(nodes, edges)
    export_vol_scc = pd.Series(
        {u: sum(d['v'] for _, _, d in G_scc.out_edges(u, data=True)) for u in nodes}
    ).sort_values(ascending=False)

    # COALITION (chosen before the output of the sector, so that its members can be starred)

    if coalition_state.get('choice') is None:
        coalition_state['choice'] = ask_coalition(code, nodes, node_to_idx,
                                                  export_vol_scc, id_to_name)
    raw_choice, top_k = coalition_state['choice']
    try:
        S, coalition_label = resolve_coalition(raw_choice, top_k, nodes, node_to_idx,
                                               export_vol_scc, id_to_name)
    except (CoalitionNameError, ValueError) as exc:
        print(f"\n [ERROR] Sector {code}, pruning {pct_frac(fraction)}: the coalition cannot be "
              f"built in this network ({exc}). This case is skipped.")
        return None

    print("\n" + "#" * 120)
    print(f"SECTOR {code} | PRUNING {pct_frac(fraction)}")
    print("#" * 120)
    print(f"\nANALYSIS FOR SECTOR: {desc}")
    print(f" Baseline year: {baseline_year} | Years used for sigma_ij: "
          f"{years[0]}-{years[-1]} ({len(years)} years)")
    print(f" PRUNING (a = {pct_frac(fraction)} of the mean baseline flow):")
    print(f"   Mean baseline flow: {prune_info['mean_flow']:,.3f} ({prune_info['mean_flow']:.3e})")
    print(f"   Threshold a = {pct_frac(fraction)} x mean flow = {prune_info['threshold']:,.3f} "
          f"({prune_info['threshold']:.3e}); arcs with flow below a are removed")
    print(f"   Arcs kept: {prune_info['arcs_after']}/{prune_info['arcs_before']} "
          f"= {pct(prune_info['arcs_after'] / prune_info['arcs_before'])}")
    print(f"   Flow kept: {prune_info['flow_after']:.3e}/{prune_info['flow_before']:.3e} "
          f"= {pct(prune_info['flow_after'] / prune_info['flow_before'])}")
    print(f" Countries in SCC: {n_n} | Flows: {n_e}")
    print(f" SCC density: {nx.density(G_scc):.2f}")
    print(f" rho^0 = {rho0:.4f}")
    print_centrality_statistics(c0, nodes, id_to_name)
    print_c0_table(c0, S, nodes, id_to_name)

    print_top_tables(c0, S, nodes, id_to_name, export_vol_scc, node_to_idx)

    print(f"\n Coalition S ({len(S)} member(s), {coalition_label}): "
          f"{', '.join(country_name(i, nodes, id_to_name) for i in S)}")

    # FEASIBILITY DATA AND VOLATILITY

    T0, ell, lambda_frac = coalition_feasibility_data(S, out_neighbors, weight0)
    d_S = coalition_indegree(S, out_neighbors, n_n)
    T_tilde = residual_budget(S, T0, ell)

    arcs = [(i, j) for i in S for j in out_neighbors[i]]
    pairs = [(nodes[i], nodes[j]) for (i, j) in arcs]
    sigma_values = arc_volatility(hist_df, code, years, pairs)
    sigma_ij = {arc: float(s) for arc, s in zip(arcs, sigma_values)}
    M_raw = import_totals_scc(edges)

    # THETA: computed once, from the baseline and the coalition
    theta, theta_min, theta_binding, theta_i = compute_theta(S, out_neighbors, nodes,
                                                             weight0, M_raw)

    # TARGET (experiment 1: c^des = c0; experiment 2: random c^des around c0) AND
    # IMPLEMENTABILITY TEST

    perturbed = []
    if experiment == 1:
        case_seed = None
        c_des = np.array(c0, dtype=float)
        print(f"\n EXPERIMENT 1: c^des = c0 (no random draw: seed not applicable)")
    elif experiment == 3:
        case_seed = None
        c_des, perturbed = generate_exp3_c_des(c0, nodes, id_to_name, export_vol_scc,
                                               node_to_idx, exp3_case, S)
        case_text = (f"the centrality of the top {EXP3_TOP_K} exporters among the members of S is decreased by "
                     f"{EXP3_FRACTION:.0%} and the mass lost is redistributed over all the other "
                     f"countries of the SCC, proportionally to c0" if exp3_case == 1 else
                     f"the centrality of Italy is increased by {EXP3_FRACTION:.0%} and the mass added is "
                     f"taken from all the other countries of the SCC, proportionally to c0")
        print(f"\n EXPERIMENT 3 - CASE {exp3_case}: c^des = c0 perturbed: {case_text} "
              f"(no random draw: seed not applicable)")
    else:
        case_seed = SEED
        c_des = generate_random_c_des(c0, nodes, case_seed)
        print(f"\n EXPERIMENT 2 - RANDOM TARGET: seed = {case_seed} (the same for every sector; "
              f"the factor of each country is drawn from default_rng([seed, country code])) | "
              f"c^des_i ~ U[c0_i (1-{DELTA_CDES:g}), c0_i (1+{DELTA_CDES:g})], then normalized")
    implementable, prop3_ok, outsiders_ok, outsider_res = check_implementable_by_S(
        c_des, S, out_neighbors, ell, T_tilde, A0, rho0)
    eta = 0.5 * min(1.0 / n_n, float(np.min(c0)))
    tag_base = f"{code}_p{round(fraction * 10000)}"

    if experiment == 1:
        c_target = c_des
        target_kind = 'c0'
        target_text = "c^target = c0 (experiment 1: c^des = c0)"
        print(f"\n c_des = c0 implementable by S: {implementable} (Proposition 3: {prop3_ok}, "
              f"outsider rows: {outsiders_ok}, residual {outsider_res:.2e})")
        print(" MIQP: not run (experiment 1: c^target = c0); only the status of the QP is reported.")
    elif implementable:
        print("\n c_des e' implementabile da S (Proposizione 3 e condizione sugli "
              "outsider soddisfatte).")
        print(" MIQP: not run (c^des is implementable by S, so c^target = c^des).")
        c_target = c_des
        target_kind = 'cdes'
        target_text = "c^target = c^des (the desired target is implementable by S)"
    else:
        failed = []
        if not prop3_ok:
            failed.append("Proposizione 3")
        if not outsiders_ok:
            failed.append(f"condizione sugli outsider (residuo {outsider_res:.2e})")
        print(f"\n c_des NON e' implementabile da S ({' e '.join(failed)} non soddisfatta).")
        c_target, stage1_obj = solve_target_centrality_miqp(
            S, out_neighbors, T0, ell, T_tilde, A0, rho0, c_des, n_n, eta,
            c_init=(None if experiment in (2, 3) else c0), verbose=False, tag=f"{tag_base}_miqp"
        )
        if experiment in (2, 3):
            print(f" MIQP started WITHOUT warm start (c_init = None, experiment {experiment}) | "
                  f"||c0 - c_des||^2 = {float(np.sum((np.asarray(c0) - np.asarray(c_des)) ** 2)):.6g} | "
                  f"||c* - c0||^2 = {float(np.sum((np.asarray(c_target) - np.asarray(c0)) ** 2)):.6g}")
        print(f" Stage 1 (target-centrality MIQP): ||c* - c_des||^2 = "
              f"{stage1_obj:.6g} | eta = {eta:.6g}")
        target_kind = 'cstar'
        target_text = ("c^target = c^* (solution of the target-centrality MIQP, "
                       "closest implementable vector to c^des)")
    print(f"\n TARGET USED IN THE COALITION GAME: {target_text}")
    dist_target = float(np.linalg.norm(np.asarray(c_target) - np.asarray(c_des)))
    print(f" ||c^target - c^des||_2 = {dist_target:.6e}"
          f"{'' if target_kind == 'cstar' else ' (c^target = c^des)'}")

    # CHECK THAT THE TARGET IS IMPLEMENTABLE BY S (Proposition 3 and outsider rows)
    target_ok, target_prop3_ok, target_outsiders_ok, target_residual = check_implementable_by_S(
        c_target, S, out_neighbors, ell, T_tilde, A0, rho0)
    print(f" CHECK c^target implementable by S: {target_ok} | Proposition 3: {target_prop3_ok} "
          f"| outsider rows: {target_outsiders_ok} (relative residual {target_residual:.2e})")
    if not target_ok:
        print("[WARNING] c^target is NOT implementable by S within the tolerances "
              f"(Prop. 3 tolerance {PROP3_TOL:g}, outsider tolerance {OUTSIDER_TOL:g}): "
              "the coalition game below may be infeasible or inaccurate.")

    print_target_diagnostics(S, nodes, id_to_name, out_neighbors, ell, T_tilde, A0,
                             rho0, c0, c_target)
    if experiment == 3:
        print_exp3_table(c0, c_des, c_target, perturbed, S, nodes, id_to_name, exp3_case)

    return {
        'code': code, 'desc': desc, 'fraction': fraction, 'tag_base': tag_base,
        'prune_info': prune_info, 'nodes': nodes, 'id_to_name': id_to_name, 'S': S,
        'out_neighbors': out_neighbors, 'weight0': weight0, 'in_from_all': in_from_all,
        'T0': T0, 'ell': ell, 'lambda_frac': lambda_frac, 'd_S': d_S, 'sigma_ij': sigma_ij,
        'M_raw': M_raw, 'rho0': rho0, 'c_target': c_target,
        'theta': theta, 'c0': c0, 'seed': case_seed, 'dist_target': dist_target,
        'experiment': experiment, 'exp3_case': exp3_case, 'c_des': c_des,
        'target_kind': target_kind, 'perturbed': perturbed,
        'c_star': c_target if target_kind == 'cstar' else None,
    }


# PROBLEM 1 FOR ONE CASE AND ONE COEFFICIENT CONFIGURATION

def calibrate_theta_equilibrium(case, configure):
    """theta_0 = 1.5 max_i theta_i^* (baseline positivity). For t = 0, 1, ...: the coalition game is
       solved with theta_t; if u_i(w^NE; theta_t) > 0 for every member i the procedure stops,
       otherwise theta_{t+1} = THETA_EQ_STEP * theta_t. Since u_i(w; theta) = theta A_i(w) - R_i(w)
       with A_i > 0 and R_i bounded on the compact set of the feasible flows, the procedure
       stops after finitely many steps. The intermediate solves are silent."""
    S, nodes, id_to_name = case['S'], case['nodes'], case['id_to_name']
    out_neighbors, weight0, in_from_all = case['out_neighbors'], case['weight0'], case['in_from_all']
    theta = case['theta']
    history = []
    for t in range(THETA_EQ_MAX_ITER + 1):
        with contextlib.redirect_stdout(io.StringIO()):
            alpha_j, beta_ij, gamma_j, _, _ = configure(
                S, out_neighbors, nodes, weight0, case['M_raw'], id_to_name,
                case['sigma_ij'], case['d_S'], theta)
            w_star, P_S_opt, _ = solve_coalition_game(
                S, out_neighbors, weight0, in_from_all, case['T0'], case['ell'],
                case['c_target'], case['rho0'], case['d_S'],
                gamma_j, alpha_j, beta_ij, verbose=False,
                tag=f"{case['tag_base']}_theta{t}",
                member_names={i: country_name(i, nodes, id_to_name) for i in S})
        u_ne = [member_utility(i, w_star, S, out_neighbors, weight0, in_from_all,
                               gamma_j, alpha_j, beta_ij) for i in S]
        n_neg = sum(1 for u in u_ne if u <= 0)
        history.append((t, theta, min(u_ne), n_neg))
        if n_neg == 0:
            return theta, history
        theta *= THETA_EQ_STEP
    raise RuntimeError(f"theta calibration on the equilibrium did not converge in "
                       f"{THETA_EQ_MAX_ITER} steps (last theta = {theta:.3e})")


# PLOTS OF THE RUN: the images are made by a separate Python process (the plotting code below), because
# matplotlib/numpy crash inside the process in which Gurobi is loaded (Windows DLL conflict).
PLOT_CODE = r'''import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 150})

LINTHRESH_PCT = 1.0          # linear region of the symmetric log scale of the utilities (in %)
ZERO_TOL = 1e-9              # a multiplier is "zero" if |m| <= ZERO_TOL * max |m| (unique multipliers)



def _title(run):
    desc = f" - {run['description']}" if run.get("description") else ""
    return f"{run['label']}{desc}"


def plot_flows(run):
    arcs = run["arcs"]
    w0 = np.array([a["w0"] for a in arcs])
    wne = np.array([a["wNE"] for a in arcs])
    ell = np.array([a["ell"] for a in arcs])
    at_floor = wne <= ell * (1 + 0.01)
    allw = np.concatenate([w0, wne])
    lo, hi = allw.min() * 0.7, allw.max() * 1.4
    grid = np.array([lo, hi])
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot(grid, grid, color="black", lw=0.8, label=r"$w^{NE}=w^0$")
    ax.plot(grid, run["lambda_frac"] * grid, color="tab:red", lw=0.8, ls="--",
            label=r"lower bound $\ell=%.1f\,w^0$" % run["lambda_frac"])
    highlight = np.zeros(len(arcs), dtype=bool)
    if run.get("key") == "exp3_case2":
        highlight = np.array([a["exporter"] in set(run.get("perturbed_members", [])) for a in arcs])
    mid = ~at_floor & ~highlight
    low = at_floor & ~highlight
    ax.scatter(w0[mid], wne[mid], s=9, color="tab:blue", alpha=0.7, linewidths=0, label="arcs")
    ax.scatter(w0[low], wne[low], s=14, color="tab:red", alpha=0.9, linewidths=0,
               label="arcs at the lower bound")
    if highlight.any():
        sc = ax.scatter(w0[highlight], wne[highlight], s=14, color="tab:green", alpha=0.9, linewidths=0,
                        label="arcs of Italy")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(f"{_title(run)}\n({len(arcs)} arcs, {int(at_floor.sum())} at the lower bound)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, which="major", lw=0.3, alpha=0.5)
    ax.set_xlabel(r"baseline flow $w^0_{ij}$ (thousands of USD)")
    ax.set_ylabel(r"equilibrium flow $w^{NE}_{ij}$ (thousands of USD)")
    first = ax.legend(handles=[h for h, l in zip(*ax.get_legend_handles_labels()) if l != "arcs of Italy"],
                      loc="upper left", frameon=False)
    if highlight.any():
        ax.add_artist(first)
        ax.legend(handles=[sc], loc="lower right", frameon=False)
    fig.tight_layout()
    return fig


def plot_exports(run, min_share=0.05):
    """One bar per member (ordered by decreasing total export T, constant by construction). Each bar is 100% of T
    and is split in two half-columns: baseline (left) and equilibrium (right). In each half-column the
    destinations are stacked by decreasing share; only the importers whose share is at least min_share (5%) at the
    baseline or at the equilibrium are named (same importer = same colour in the two halves), the rest is grey ('others')."""
    exporters = {}
    for a in run["arcs"]:
        exporters.setdefault(a["exporter"], []).append(a)
    T = {e: sum(a["w0"] for a in arcs) for e, arcs in exporters.items()}
    iso3 = {a["importer"]: a.get("importer_iso3", a["importer"]) for a in run["arcs"]}
    order = sorted(exporters, key=lambda e: -T[e])
    palette = [plt.cm.tab10(k) for k in range(10) if k != 7] + [plt.cm.Set3(k) for k in range(12)]
    fig, ax = plt.subplots(figsize=(22, 9))
    half = 0.42
    for pos, e in enumerate(order):
        arcs = exporters[e]
        total0 = sum(a["w0"] for a in arcs)
        totalN = sum(a["wNE"] for a in arcs)
        share0 = {a["importer"]: a["w0"] / total0 for a in arcs}
        shareN = {a["importer"]: a["wNE"] / totalN for a in arcs}
        named = sorted((d for d in share0 if share0[d] >= min_share or shareN[d] >= min_share),
                       key=lambda d: -max(share0[d], shareN[d]))
        top0 = named
        topN = named
        colors = {d: palette[k % len(palette)] for k, d in enumerate(named)}
        for side, shares, tops in ((0, share0, top0), (1, shareN, topN)):
            x0 = pos - half if side == 0 else pos
            bottom = 0.0
            for d in sorted(tops, key=lambda d: -shares[d]):
                h = shares[d]
                ax.bar(x0, h, width=half, bottom=bottom, align="edge", color=colors[d],
                       edgecolor="white", linewidth=0.4)
                ax.text(x0 + half / 2, bottom + h / 2, iso3[d], ha="center", va="center",
                        fontsize=min(6.0, max(3.0, 450.0 * h)))
                bottom += h
            rest = 1.0 - bottom
            if rest > 1e-9:
                ax.bar(x0, rest, width=half, bottom=bottom, align="edge", color="lightgrey",
                       edgecolor="white", linewidth=0.4)
                ax.text(x0 + half / 2, bottom + rest / 2, "others", ha="center", va="center",
                        fontsize=min(5.0, max(3.0, 450.0 * rest)))
            ax.text(x0 + half / 2, 1.01, "B" if side == 0 else "E", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{e}  (T = {T[e]:,.0f})" for e in order], rotation=90, fontsize=8)
    ax.set_xlim(-0.8, len(order) - 0.2)
    ax.set_ylim(0, 1.06)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("share of the total export of the member")
    ax.set_title(f"{_title(run)} - export shares, baseline (B) and equilibrium (E); "
                 f"T = total export (thousands of USD)")
    fig.tight_layout()
    return fig


def plot_utilities(run):
    names = [u["country"] for u in run["utilities"]]
    vals = np.array([100.0 * (u["uNE"] - u["u0"]) / abs(u["u0"]) if u["u0"] != 0 else np.nan
                     for u in run["utilities"]])
    order = sorted(range(len(names)), key=lambda k: names[k])
    names = [names[k] for k in order]
    vals = vals[order]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    bars = ax.bar(x, vals, color=np.where(vals >= 0, "tab:green", "tab:red"), width=0.75)
    perturbed = set(run.get("perturbed_members", []))
    for name, bar in zip(names, bars):
        if name in perturbed:
            bar.set_hatch("///")
            bar.set_edgecolor("black")
            bar.set_linewidth(0.8)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_yscale("symlog", linthresh=LINTHRESH_PCT)
    fin = vals[np.isfinite(vals)]
    ax.set_ylim(min(fin.min(), -1) * 1.3, max(fin.max(), 1) * 1.3)
    ax.set_title(_title(run))
    ax.grid(True, axis="y", which="major", lw=0.3, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90)
    ax.set_ylabel(r"$\Delta u_i$ (%), symmetric log scale")
    if perturbed:
        ax.legend(handles=[Patch(facecolor="white", edgecolor="black", hatch="///",
                                 label="perturbed member(s) of the coalition")],
                  loc="best", frameon=False)
    fig.tight_layout()
    return fig


json_path, folder = sys.argv[1], sys.argv[2]
with open(json_path, encoding="utf-8") as fh:
    run = json.load(fh)
figs = [("fig_flows", plot_flows(run)), ("fig_utilities", plot_utilities(run)),
        ("fig_exports", plot_exports(run))]
for name, fig in figs:
    path = os.path.join(folder, f"{name}_{run['key']}.jpg")
    fig.savefig(path, bbox_inches="tight", dpi=200, pil_kwargs={"quality": 95})
    print(path)
'''


def make_outputs(run, folder):
    import subprocess
    json_path = os.path.join(folder, f"plot_data_{run['key']}.json")
    # when the script is run from VS Code the conda environment is not activated: its DLL folders (MKL, used by
    # numpy) are missing from PATH and numpy.linalg crashes (error 0xc06d007f). They are added here.
    prefix = os.path.dirname(os.path.abspath(sys.executable))
    extra = [prefix] + [os.path.join(prefix, "Library", d) for d in ("bin", "mingw-w64\\bin", "usr\\bin")] \
        + [os.path.join(prefix, "Scripts")]
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([d for d in extra if os.path.isdir(d)] + [env.get("PATH", "")])
    result = subprocess.run([sys.executable, "-c", PLOT_CODE, json_path, folder],
                            capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"plotting process failed (code {result.returncode}): {result.stderr.strip()[-1500:]}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def save_plot_data(case, S, nodes, id_to_name, out_neighbors, weight0, w_star, u_base, u_ne, theta,
                   multipliers=None, nu=None, certificate=None):
    """Saves the data needed by the plots (flows and utilities of the members) in a JSON file
    placed next to this script: plot_data_exp1.json, plot_data_exp2.json, plot_data_exp3_case1.json,
    plot_data_exp3_case2.json."""
    experiment, exp3_case = case['experiment'], case['exp3_case']
    if experiment == 3:
        key, label = f"exp3_case{exp3_case}", f"Experiment 3.{exp3_case}"
    else:
        key, label = f"exp{experiment}", f"Experiment {experiment}"
    S_set = set(S)
    try:
        iso_df = pd.read_csv(COUNTRIES_FILE, usecols=['country_code', 'country_iso3'])
        code_to_iso3 = dict(zip(iso_df['country_code'].astype(int), iso_df['country_iso3'].astype(str)))
    except Exception:
        code_to_iso3 = {}
    arcs = []
    for i in S:
        for j in out_neighbors[i]:
            arcs.append({'exporter': country_name(i, nodes, id_to_name),
                         'importer': country_name(j, nodes, id_to_name),
                         'importer_iso3': code_to_iso3.get(int(nodes[j]), country_name(j, nodes, id_to_name)),
                         'importer_in_S': bool(j in S_set),
                         'w0': float(weight0[(i, j)]), 'wNE': float(w_star[i, j]),
                         'ell': float(case['ell'][i][j])})
    utilities = [{'country': country_name(i, nodes, id_to_name),
                  'u0': float(u0), 'uNE': float(uN)}
                 for i, u0, uN in zip(S, u_base, u_ne)]
    perturbed_members = [country_name(k, nodes, id_to_name) for k in case.get('perturbed', []) if k in S_set]
    if experiment == 3 and exp3_case == 1:
        description = f"top-{EXP3_TOP_K} EU exporters, -{EXP3_FRACTION:.0%}"
    elif experiment == 3:
        description = f"Italy, +{EXP3_FRACTION:.0%}"
    else:
        description = ""
    mult_data = None
    if multipliers is not None and nu is not None:
        members = []
        for i in S:
            unique = certificate[i]['unique']      # Prop. 4 (valid KKT) and Prop. 5 (LICQ)
            members.append({'country': country_name(i, nodes, id_to_name), 'out_degree': len(out_neighbors[i]),
                            'mu': float(multipliers['mu'][i]), 'pi': float(multipliers['pi'][i]),
                            'unique': bool(unique)})
        arcs_mult = [{'exporter': country_name(i, nodes, id_to_name), 'importer': country_name(j, nodes, id_to_name),
                      'nu': float(nu[(i, j)]),
                      'active': bool(w_star[i, j] - case['ell'][i][j] <= LB_TOL * case['ell'][i][j])}
                     for i in S for j in out_neighbors[i]]
        mult_data = {'members': members, 'arcs': arcs_mult}
    data = {'key': key, 'label': label, 'description': description, 'perturbed_members': perturbed_members,
            'sector': case['code'], 'theta': float(theta),
            'lambda_frac': float(case['lambda_frac']), 'arcs': arcs, 'utilities': utilities,
            'multipliers': mult_data}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"plot_data_{key}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    return data


def print_multipliers_table(plot_data):
    """Prints, in the run output, the summary table of the multipliers mu_i, pi_i, nu_ij. Only the UNIQUE
       multipliers enter the statistics (members for which the KKT system (22)-(27) is verified, Prop. 4,
       and the LICQ condition holds, Prop. 5); the others are counted in the column 'Not unique'."""
    mult = plot_data.get('multipliers')
    if not mult:
        return
    mem, arcs = mult['members'], mult['arcs']
    uniq = {m['country'] for m in mem if m['unique']}
    rows = []
    for label, key in (("mu_i (budget)", 'mu'), ("pi_i (centrality)", 'pi')):
        vals = [(m[key], m['country']) for m in mem if m['unique']]
        scale = max([abs(v) for v, _ in vals] + [1.0])
        nonzero = [v for v, _ in vals if abs(v) > 1e-9 * scale]
        rows.append((label, len(mem), len(nonzero), len(vals) - len(nonzero), len(mem) - len(uniq),
                     min(vals) if vals else None, max(vals) if vals else None))
    a_uniq = [a for a in arcs if a['exporter'] in uniq]
    act = [a for a in a_uniq if a['active']]
    ref = [(a['nu'], f"{a['exporter']} -> {a['importer']}") for a in act]
    rows.append(("nu_ij (lower bound)", len(arcs), len(act), len(a_uniq) - len(act), len(arcs) - len(a_uniq),
                 min(ref) if ref else None, max(ref) if ref else None))

    def cell(entry):
        return "-" if entry is None else f"{entry[0]:.6e} ({entry[1]})"
    width = 150
    print("\n" + "=" * width)
    print(f"MULTIPLIERS TABLE - {plot_data['label']}")
    print("=" * width)
    print(f" {'Multiplier':<22}{'Constraints':>12}{'Active':>8}{'Non-active':>12}{'Not unique':>12}"
          f"   {'Minimum (reference)':<48}{'Maximum (reference)'}")
    print("-" * width)
    for label, n, n_act, n_non, n_nu, lo, hi in rows:
        print(f" {label:<22}{n:>12d}{n_act:>8d}{n_non:>12d}{n_nu:>12d}   {cell(lo):<48}{cell(hi)}")
    print("-" * width)
    print(f" Active: mu_i, pi_i nonzero; nu_ij arc at the lower bound. Not unique: members for which uniqueness is "
          f"not guaranteed (Prop. 5 fails or Prop. 4 not verified)\n (excluded from all the other columns). Minimum/maximum over the "
          f"unique multipliers ({len(uniq)}/{len(mem)} members, {len(a_uniq)}/{len(arcs)} arcs);"
          f" for nu_ij over the active arcs.")
    print("=" * width)


def run_game(case, index, configure):
    S, nodes, id_to_name = case['S'], case['nodes'], case['id_to_name']
    out_neighbors, weight0, in_from_all = case['out_neighbors'], case['weight0'], case['in_from_all']
    prune_info = case['prune_info']
    row = {'config': f"Config. {index}", 'fraction': case['fraction'], 'error': None,
           'theta': case['theta'], 'dist_target': case['dist_target'],
           'poa': None, 'W_NE': None, 'W_SO': None,
           'threshold': prune_info['threshold'],
           'arcs_after': prune_info['arcs_after'], 'arcs_before': prune_info['arcs_before'],
           'arcs_ratio': prune_info['arcs_after'] / prune_info['arcs_before'],
           'flow_ratio': prune_info['flow_after'] / prune_info['flow_before']}
    print("\n" + "*" * 120)
    print(f"SECTOR {case['code']} | PRUNING {pct_frac(case['fraction'])}")
    print("*" * 120)
    try:
        theta_0 = case['theta']
        theta, theta_history = calibrate_theta_equilibrium(case, configure)
        row['theta'] = theta
        print(f"\n THETA CALIBRATION ON THE EQUILIBRIUM (theta_0 = 1.5 max_i theta_i^* = {theta_0:.6e}; "
              f"theta is multiplied by {THETA_EQ_STEP:g} until all the equilibrium utilities are positive)")
        print(f" {'step t':>7}{'theta_t':>16}{'min_i u_i(NE)':>22}{'members with u_i<=0':>22}")
        for t, th, min_u, n_neg in theta_history:
            print(f" {t:>7d}{th:>16.6e}{min_u:>22.6e}{n_neg:>22d}")
        alpha_j, beta_ij, gamma_j, label, details = configure(
            S, out_neighbors, nodes, weight0, case['M_raw'], id_to_name,
            case['sigma_ij'], case['d_S'], theta)
        print(f"\n THETA: theta = {theta:.6e} (theta / theta_0 = {theta / theta_0:.4f})")
        print_k_check(S, out_neighbors, case['sigma_ij'], beta_ij, gamma_j, case['d_S'])
        dominant, violations = check_diagonal_dominance(
            S, out_neighbors, beta_ij, case['d_S'], gamma_j)
        print(f"\n condition (Prop. 2) satisfied: {dominant} | gamma_j = {GAMMA_VALUE:g} | "
              f"ell_ij = {case['lambda_frac']:.2%} * w0_ij")

        w_star, P_S_opt, multipliers = solve_coalition_game(
            S, out_neighbors, weight0, in_from_all, case['T0'], case['ell'],
            case['c_target'], case['rho0'], case['d_S'],
            gamma_j, alpha_j, beta_ij, verbose=False,
            tag=f"{case['tag_base']}_config{index}",
            member_names={i: country_name(i, nodes, id_to_name) for i in S}
        )
        nu = compute_lower_bound_multipliers(
            S, out_neighbors, weight0, in_from_all, w_star, case['c_target'],
            gamma_j, alpha_j, beta_ij, multipliers)
        print_results(S, id_to_name, out_neighbors, weight0, in_from_all, nodes,
                      w_star, P_S_opt, gamma_j, alpha_j, beta_ij, dominant, multipliers)
        row['n_active'] = print_lower_bound_multipliers(
            S, id_to_name, nodes, out_neighbors, nu, w_star, case['ell'], case['T0'])

        w0_dict = {(i, j): weight0[(i, j)] for i in S for j in out_neighbors[i]}
        u_base = [member_utility(i, w0_dict, S, out_neighbors, weight0, in_from_all,
                                 gamma_j, alpha_j, beta_ij) for i in S]
        u_ne = [member_utility(i, w_star, S, out_neighbors, weight0, in_from_all,
                               gamma_j, alpha_j, beta_ij) for i in S]
        base_pos, base_neg = count_signs(u_base)
        ne_pos, ne_neg = count_signs(u_ne)
        try:
            certificate = verify_multipliers(S, out_neighbors, w_star, case['c_target'], case['ell'],
                                             case['T0'], case['rho0'], nu, multipliers, P_S_opt)
            for i in S:
                if certificate[i]['licq'] and not certificate[i]['valid']:
                    print(f"[WARNING] {country_name(i, nodes, id_to_name)}: LICQ holds (Prop. 5) but the KKT "
                          f"system (22)-(27) is not verified within tolerance (Prop. 4)")
        except Exception as exc:
            certificate = None
            print(f"[WARNING] Multipliers verification not performed: {type(exc).__name__}: {exc}")
        try:
            plot_data = save_plot_data(case, S, nodes, id_to_name, out_neighbors, weight0, w_star,
                                       u_base, u_ne, theta, multipliers, nu, certificate)
        except Exception as exc:
            plot_data = None
            print(f"[WARNING] Plot data not saved: {type(exc).__name__}: {exc}")
        if plot_data is not None:
            try:
                print_multipliers_table(plot_data)
            except Exception as exc:
                print(f"[WARNING] Multipliers table not printed: {type(exc).__name__}: {exc}")
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                print("\n Creating the images...", flush=True)
                saved_files = make_outputs(plot_data, script_dir)
                print("\n" + "#" * 100)
                print(" IMAGES CREATED (flows, utilities, export shares):", flush=True)
                for saved in saved_files:
                    print(f"   {saved}")
                print("#" * 100)
                for saved in saved_files:
                    try:
                        os.startfile(saved)
                    except Exception:
                        pass
            except Exception as exc:
                import traceback
                print("\n" + "#" * 100)
                print(f"[WARNING] IMAGES NOT PRODUCED: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                print("#" * 100)
        row.update({'potential': P_S_opt, 'base_pos': base_pos, 'base_neg': base_neg,
                    'ne_pos': ne_pos, 'ne_neg': ne_neg})
        print(f"\n Utilities at the baseline: positive = {base_pos}, negative = {base_neg}")
        print(f" Utilities at the equilibrium: positive = {ne_pos}, negative = {ne_neg}")

        poa = price_of_anarchy(S, out_neighbors, weight0, in_from_all, case['T0'], case['ell'],
                               case['c_target'], case['rho0'], case['d_S'],
                               gamma_j, alpha_j, beta_ij, w_star)
        row.update({'poa': poa['poa'], 'W_NE': poa['W_NE'], 'W_SO': poa['W_SO']})
        if poa['poa'] is not None:
            print(f"\n Price of Anarchy = W(w^SO)/W(w^NE) = {poa['poa']:.6f}")
        else:
            print(f"\n Price of Anarchy = W(w^SO)/W(w^NE): n.d. ({poa['note']})")
    except Exception as exc:
        row['error'] = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] Case stopped: {row['error']}")
    return row


def ask_experiment():
    while True:
        raw = input("Enter experiment (1 = c^des = c0, coalition game only; "
                    "2 = random c^des around c0, MIQP if needed; "
                    "3 = deterministic perturbation of c0, MIQP if needed): ").strip()
        if raw in ("1", "2", "3"):
            print(f"\nEXPERIMENT {raw} selected.")
            variant = None
            if raw == "3":
                while variant not in ("1", "2"):
                    variant = input(f"Enter case of experiment 3 (1 = centrality of the top {EXP3_TOP_K} "
                                    f"exporters among the members of S decreased by {EXP3_FRACTION:.0%}, mass redistributed "
                                    f"proportionally to c0; 2 = centrality of Italy increased by "
                                    f"{EXP3_FRACTION:.0%}): ").strip()
                variant = int(variant)
                print(f"\nEXPERIMENT 3, CASE {variant} selected.")
            return int(raw), variant
        print(" Please enter 1, 2 or 3.")


def main():
    sector_descriptions = load_sector_descriptions(SECTOR_DESCRIPTION_FILE)
    sector_codes = [input("Enter sector code: ").strip().zfill(6)]
    experiment, exp3_case = ask_experiment()

    hist_df, years = load_flows_all_years(sector_codes)
    baseline_year = max(years)
    flows_df = hist_df.loc[hist_df['year'] == baseline_year, ['i', 'j', 'k', 'v']]

    countries_df = pd.read_csv(COUNTRIES_FILE, usecols=['country_code', 'country_name'])
    countries_df['country_name'] = countries_df['country_name'].str.encode('latin1').str.decode('utf-8')
    id_to_name = dict(zip(countries_df['country_code'].astype(int), countries_df['country_name']))

    # STAGE A: network, coalition and target for every case (sector x pruning level)

    cases = []
    for code in sector_codes:
        desc = sector_descriptions.get(code, "Description not available")
        coalition_state = {'choice': None}
        for fraction in PRUNING_FRACTIONS:
            case = prepare_case(code, desc, fraction, flows_df, hist_df, years,
                                baseline_year, id_to_name, coalition_state, experiment, exp3_case)
            if case is not None:
                cases.append(case)

    # STAGE B: coalition game for every case

    rows_by_sector = {code: [] for code in sector_codes}
    descriptions = {case['code']: case['desc'] for case in cases}
    configure = EXPERIMENT_CONFIGURATIONS[0]
    for case in cases:
        rows_by_sector[case['code']].append(run_game(case, 1, configure))

    # EXPERIMENT SUMMARIES (one per sector)

    for code in sector_codes:
        if rows_by_sector[code]:
            print_experiment_summary(code, descriptions.get(code, ""), rows_by_sector[code],
                                     experiment)

    # c0 AND TARGET (once, at the end of the output)

    print_final_vectors(cases)


if __name__ == "__main__":
    output_log, original_streams = configure_output_log()
    try:
        main()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout, sys.stderr = original_streams
        faulthandler.disable()
        output_log.close()
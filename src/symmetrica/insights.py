"""Automated tactical insights: turn the metric summary into short, checkable sentences.

Every insight is produced by an explicit rule on numbers that appear
elsewhere in the report and carries the evidence it is based on, plus a
`strength` in [0, 1] saying how one-sided the comparison is. Comparisons
closer than `EVEN` are reported as evenly matched rather than as a win.
Insights come out in a fixed tactical order (space, pressing, progression,
players, structure, momentum). Nothing is generated free-form, so every
sentence can be traced back to a number.
"""
from dataclasses import asdict, dataclass

import math

NAMES = {"home": "Home", "away": "Away"}
EVEN = 0.1  # strength below which a comparison counts as evenly matched


@dataclass
class Insight:
    category: str
    team: str | None
    headline: str
    detail: str
    strength: float

    def to_dict(self) -> dict:
        return asdict(self)


def _player(pid: str | None) -> str:
    return f"#{pid.split('_')[1]}" if pid else "?"


def _ratio_strength(a: float, b: float) -> float:
    """0 when a == b, approaching 1 as one side dominates."""
    if not (math.isfinite(a) and math.isfinite(b)) or a + b <= 0:
        return 0.0
    return min(abs(a - b) / (a + b) * 2, 1.0)


def _f(v) -> float:
    """None (how NaN travels through JSON) back to NaN."""
    return float("nan") if v is None else float(v)


def _cmp(summary: dict, section: str, key: str) -> tuple[float, float]:
    return _f(summary[section]["home"][key]), _f(summary[section]["away"][key])


def generate(summary: dict) -> list[Insight]:
    out: list[Insight] = []

    # Space: Voronoi dominance overall and in the final third.
    sp = summary["space"]
    share = _f(sp["home_share"])
    lead = "home" if share >= 0.5 else "away"
    lead_share = share if lead == "home" else 1 - share
    ft_h, ft_a = _f(sp["home_final_third"]), _f(sp["away_final_third"])
    strength = min(abs(share - 0.5) * 4, 1.0)  # 52.5 / 47.5 is the smallest split called a win
    out.append(Insight(
        "space", lead if strength >= EVEN else None,
        f"{NAMES[lead]} owned more of the pitch" if strength >= EVEN else "Space was shared evenly",
        f"{NAMES[lead]} was the nearest team to {lead_share:.0%} of the pitch on average (Voronoi). "
        f"In the final third they attack, Home controlled {ft_h:.0%} and Away {ft_a:.0%}.",
        strength))

    # Pressing: PPDA (lower = more aggressive).
    p_h, p_a = _cmp(summary, "pressing", "ppda")
    if math.isfinite(p_h) and math.isfinite(p_a):
        hi = "home" if p_h < p_a else "away"
        lo = "away" if hi == "home" else "home"
        even = _ratio_strength(p_h, p_a) < EVEN
        out.append(Insight(
            "pressing", None if even else hi,
            "Both teams pressed with similar intensity" if even else f"{NAMES[hi]} pressed higher and harder",
            f"PPDA Home {p_h:.1f}, Away {p_a:.1f} (passes allowed per defensive action; lower = more "
            f"aggressive)." + ("" if even else f" {NAMES[lo]} got fewer passes away in their own build-up "
                                        f"zone before {NAMES[hi]} engaged."),
            _ratio_strength(p_h, p_a)))

    # Counter-pressing.
    c_h, c_a = _cmp(summary, "pressing", "counterpress_rate")
    best = "home" if c_h >= c_a else "away"
    pr = summary["pressing"][best]
    extra = ""
    if pr.get("counterpress_pressers") is not None:
        extra = f" On average {pr['counterpress_pressers']:.1f} players were within 10 m of the ball in those 5 s."
    out.append(Insight(
        "counter-press", best, f"{NAMES[best]}'s counter-press was more effective"
        if _ratio_strength(c_h, c_a) >= EVEN else "Counter-pressing was evenly matched",
        f"{NAMES[best]} won the ball back within 5 s after {max(c_h, c_a):.0%} of open-play losses "
        f"({pr['counterpress_regains']} of {pr['turnovers']}), vs {min(c_h, c_a):.0%} for "
        f"{NAMES['away' if best == 'home' else 'home']}.{extra}",
        _ratio_strength(c_h, c_a)))

    # High turnovers.
    h_h, h_a = _cmp(summary, "pressing", "high_turnovers")
    if h_h + h_a > 0:
        t = "home" if h_h >= h_a else "away"
        pr = summary["pressing"][t]
        out.append(Insight(
            "high turnovers", t, f"{NAMES[t]} won the ball high more often",
            f"{pr['high_turnovers']} regains within 40 m of the opponent's goal "
            f"(vs {min(h_h, h_a):.0f}), {pr['high_turnover_shots']} of them ending in a shot.",
            _ratio_strength(h_h, h_a)))

    # Threat.
    x_h, x_a = _cmp(summary, "threat", "xt_generated")
    t = "home" if x_h >= x_a else "away"
    th = summary["threat"][t]
    carry_share = th["xt_carries"] / th["xt_generated"] if th["xt_generated"] else 0.0
    out.append(Insight(
        "threat", t, f"{NAMES[t]} created more threat"
        if _ratio_strength(x_h, x_a) >= EVEN else "Threat creation was evenly matched",
        f"{max(x_h, x_a):.2f} xT from passes and carries vs {min(x_h, x_a):.2f}; "
        f"{carry_share:.0%} of it came from carrying the ball.",
        _ratio_strength(x_h, x_a)))

    # Finishing against shot quality.
    for team in ("home", "away"):
        th = summary["threat"][team]
        if th["shots"]:
            diff = th["goals"] - th["xg"]
            word = "finished in line with" if abs(diff) < 0.5 else ("outperformed" if diff > 0 else "underperformed")
            out.append(Insight(
                "finishing", team, f"{NAMES[team]} {word} their chances",
                f"{th['goals']} goals from {th['shots']} shots worth {th['xg']:.2f} geometric xG.",
                min(abs(diff) / 2, 1.0)))

    # Line-breaking.
    l_h, l_a = _cmp(summary, "threat", "line_breaking_passes")
    t = "home" if l_h >= l_a else "away"
    leaders = [r for r in summary["threat"]["leaders"] if r["team"] == t]
    top_lb = max(leaders, key=lambda r: r["line_breaking"]) if leaders else None
    who = f" {_player(top_lb['player'])} led with {int(top_lb['line_breaking'])}." if top_lb else ""
    out.append(Insight(
        "line-breaking", t, f"{NAMES[t]} played through the lines more"
        if _ratio_strength(l_h, l_a) >= EVEN else "Both teams broke lines at a similar rate",
        f"{NAMES[t]}: {max(l_h, l_a):.0f} line-breaking passes vs {min(l_h, l_a):.0f}, taking out "
        f"{summary['threat'][t]['packing']} opponents in total (packing).{who}",
        _ratio_strength(l_h, l_a)))

    # Key progressor overall.
    if summary["threat"]["leaders"]:
        top = summary["threat"]["leaders"][0]
        out.append(Insight(
            "players", top["team"], f"{NAMES[top['team']]} {_player(top['player'])} was the main progressor",
            f"{top['xt']:.2f} xT added ({top['xt_passes']:.2f} passing, {top['xt_carries']:.2f} carrying), "
            f"{int(top['progressive'])} progressive passes.",
            0.6))

    # Passing structure.
    for team in ("home", "away"):
        net = summary["networks"][team]
        if net["key_link"]:
            k = net["key_link"]
            out.append(Insight(
                "network", team, f"{NAMES[team]}'s build-up ran through {_player(net['hub'])}",
                f"Most frequent combination {_player(k['a'])}-{_player(k['b'])} ({k['count']} passes); "
                f"the three busiest links carried {net['top3_link_share']:.0%} of the team's passes.",
                min(net["top3_link_share"] * 2, 1.0)))

    # Defensive block: compactness out of possession, and how the shape changes with the ball.
    sh = summary.get("shape")
    if sh:
        area = {t: _f((sh[t]["out"] or {}).get("area")) for t in ("home", "away")}
        if all(math.isfinite(v) for v in area.values()):
            t = "home" if area["home"] <= area["away"] else "away"
            o = "away" if t == "home" else "home"
            even = _ratio_strength(area["home"], area["away"]) < EVEN
            out.append(Insight(
                "shape", None if even else t,
                "Both defensive blocks were similarly compact" if even else f"{NAMES[t]} defended more compactly",
                f"Out of possession the outfield hull covered {area[t]:.0f} m² for {NAMES[t]} vs "
                f"{area[o]:.0f} m², {sh[t]['out']['width']:.0f} m wide and {sh[t]['out']['depth']:.0f} m deep, "
                f"its centroid {sh[t]['out']['height']:.0f} m from its own goal.",
                _ratio_strength(area["home"], area["away"])))
        forms = sh.get("formations", {})
        for team in ("home", "away"):
            f_in, f_out = (forms.get(team) or {}).get("in"), (forms.get(team) or {}).get("out")
            if f_in and f_out:
                same = f_in["formation"] == f_out["formation"]
                out.append(Insight(
                    "formation", team,
                    f"{NAMES[team]} played a {f_in['formation']}" if same else
                    f"{NAMES[team]}: {f_out['formation']} without the ball, {f_in['formation']} with it",
                    "Template match of the average outfield shape (Hungarian assignment); "
                    f"in-possession hull {sh[team]['in']['area']:.0f} m² vs {sh[team]['out']['area']:.0f} m² out.",
                    0.3 if same else 0.6))

    # Off-ball running and the space it created.
    ob = summary.get("offball")
    if ob:
        cat = "Behind defensive line"
        b_h, b_a = ob["home"]["by_category"][cat], ob["away"]["by_category"][cat]
        s_h, s_a = _f(ob["home"]["space_m2"]), _f(ob["away"]["space_m2"])
        t = "home" if s_h >= s_a else "away"
        o = "away" if t == "home" else "home"
        out.append(Insight(
            "off-ball runs", t, f"{NAMES[t]}'s runs created more space",
            f"{ob[t]['runs']} high-intensity off-ball runs opened {ob[t]['space_m2']:,.0f} m² of pitch control "
            f"for teammates (vs {ob[o]['space_m2']:,.0f} m²); runs in behind: Home {b_h}, Away {b_a}; "
            f"overlaps/underlaps: {ob[t]['overlaps']}/{ob[t]['underlaps']}.",
            _ratio_strength(s_h, s_a)))

    # Pass decisions.
    dec = summary.get("decisions")
    if dec and dec["home"]["passes_evaluated"] and dec["away"]["passes_evaluated"]:
        b_h, b_a = _f(dec["home"]["best_option_share"]), _f(dec["away"]["best_option_share"])
        t = "home" if b_h >= b_a else "away"
        out.append(Insight(
            "decisions", t, f"{NAMES[t]} picked the best pass option more often"
            if _ratio_strength(b_h, b_a) >= EVEN else "Passing decisions were evenly matched",
            f"Chosen receiver = highest-EPV option on {max(b_h, b_a):.0%} of passes vs {min(b_h, b_a):.0%}; "
            f"top-3 option {dec[t]['top3_share']:.0%} of the time, leaving "
            f"{dec[t]['epv_left_per_pass'] * 1000:.1f} mEPV per pass on the table.",
            _ratio_strength(b_h, b_a)))

    # Momentum: the strongest ten-minute spell, relative to the team's whole-match threat.
    for team in ("home", "away"):
        start, value = summary["momentum"]["best_spell"][team]
        total = _f(summary["threat"][team]["xt_generated"])
        if value > 0:
            out.append(Insight(
                "momentum", team, f"{NAMES[team]}'s strongest spell: {start}'-{start + 10}'",
                f"Out-created the opponent by {value:.2f} xT over those ten minutes.",
                min(value / total, 1.0) if total > 0 else 0.0))

    return out

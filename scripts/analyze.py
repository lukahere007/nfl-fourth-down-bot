"""Turn results/decisions.parquet into tables, headline numbers and figures.

Usage: python scripts/analyze.py
Needs only committed files (results/*.parquet|csv), so it also runs in CI.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
COL = {"go": "#2a78d6", "fg": "#eb6834", "punt": "#1baf7a"}
LIGHT = {"go": "#a9c8ee", "fg": "#f6c0aa", "punt": "#a3dfc9"}
LABEL = {"go": "Go for it", "fg": "Field goal", "punt": "Punt"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CLEAR_CONF, CLEAR_GAP = 0.90, 0.01

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white"})


def relevant(D: pd.DataFrame) -> pd.DataFrame:
    """Keep decision-relevant snaps: pre-snap WP 5-95% and > 30 s left in the half.

    The engine charges every 4th-down play a fixed 6 s of clock, which is too
    crude for the last few snaps of a half, where clock management dominates.
    """
    return D[D["wp_pre"].between(0.05, 0.95) & (D["half_seconds_remaining"] > 30)]


def season_table(R: pd.DataFrame) -> pd.DataFrame:
    clear = R[(R.confidence >= CLEAR_CONF) & (R.gap >= CLEAR_GAP)]
    rows = []
    for s, g in R.groupby("season"):
        c = clear[clear.season == s]
        cg = c[c.best == "go"]
        rows.append({"season": s, "fourth_downs": len(g), "go_rate": (g.decision == "go").mean(),
                     "bot_go_rate": (g.best == "go").mean(), "agree_rate": (g.best == g.decision).mean(),
                     "clear_go_situations": len(cg), "went_when_clear_go": (cg.decision == "go").mean(),
                     "wp_lost_per_game_team": g.wp_lost.sum() / g.groupby(["game_id", "posteam"]).ngroups})
    return pd.DataFrame(rows)


def fig_decision_chart(chart: pd.DataFrame) -> None:
    acts = ["go", "fg", "punt"]
    scen = list(dict.fromkeys(chart.scenario))
    fig, axes = plt.subplots(len(scen), 1, figsize=(10, 3.4 * len(scen)), constrained_layout=True)
    for ax, sc in zip(np.atleast_1d(axes), scen):
        c = chart[chart.scenario == sc]
        Z = np.full((15, 99), np.nan)
        for r in c.itertuples():
            code = acts.index(r.best) * 2 + (1 if r.gap < CLEAR_GAP else 0)
            Z[r.ydstogo - 1, r.yardline_100 - 1] = code
        cmap = ListedColormap([COL["go"], LIGHT["go"], COL["fg"], LIGHT["fg"], COL["punt"], LIGHT["punt"]])
        ax.imshow(Z[:, ::-1], origin="lower", aspect="auto", cmap=cmap, vmin=-0.5, vmax=5.5,
                  extent=[0.5, 99.5, 0.5, 15.5], interpolation="nearest")
        ax.set_xticks([9.5, 29.5, 49.5, 69.5, 89.5], ["Own 10", "Own 30", "50", "Opp 30", "Opp 10"])
        ax.set_yticks([1, 3, 5, 10, 15])
        ax.set_ylabel("Yards to go")
        ax.set_title(sc, loc="left", color=INK, fontsize=11)
        ax.spines[:].set_visible(False)
    handles = [Patch(color=COL[a], label=f"{LABEL[a]}") for a in acts] + \
              [Patch(color="#d9d9d6", label=f"lighter = margin < {CLEAR_GAP:.0%} WP")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Bot's best 4th-down call (pick'em game, 2025 kicking & kickoff conventions)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.savefig(FIG / "decision_chart.png", dpi=150)
    plt.close(fig)


def fig_trend(st: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    ax.plot(st.season, st.went_when_clear_go * 100, color=COL["go"], lw=2, marker="o", ms=6)
    ax.plot(st.season, st.go_rate * 100, color=MUTED, lw=2, marker="o", ms=6)
    ax.text(st.season.iloc[-1] + 0.2, st.went_when_clear_go.iloc[-1] * 100,
            "Went for it when the bot\nclearly said go", color=INK, va="center", fontsize=9)
    ax.text(st.season.iloc[-1] + 0.2, st.go_rate.iloc[-1] * 100, "All 4th downs: go rate",
            color=INK, va="center", fontsize=9)
    ax.set_xlim(st.season.min() - 0.5, st.season.max() + 3.2)
    ax.set_xticks(st.season[::2])
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of 4th downs (pre-snap WP 5-95%)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_title("Coaches have become more aggressive, but still leave many clear 'go' calls on the table",
                 loc="left", fontsize=11, color=INK)
    fig.savefig(FIG / "aggressiveness_trend.png", dpi=150)
    plt.close(fig)


def fig_calibration(cal: pd.DataFrame) -> None:
    phases = ["Q1", "Q2", "Q3", "Q4 (>5 min)", "Last 5 min"]
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.9), sharey=True, constrained_layout=True)
    for ax, ph in zip(axes, phases):
        ax.plot([0, 1], [0, 1], color=GRID, lw=1.5)
        for m, col, lab in [("logistic", MUTED, "Logistic baseline"), ("xgb_mono", COL["go"], "Monotone XGBoost")]:
            t = cal[(cal.phase == ph) & (cal.model == m)]
            ax.plot(t.pred, t.obs, color=col, lw=2, marker="o", ms=4, label=lab)
        ll = cal[(cal.phase == ph)].groupby("model").log_loss.first()
        ax.set_title(f"{ph}\nlog loss: XGB {ll['xgb_mono']:.3f}, logistic {ll['logistic']:.3f}", fontsize=9, color=INK)
        ax.set_xlabel("Predicted WP")
        ax.set_aspect("equal")
    axes[0].set_ylabel("Observed win rate")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("WP model calibration on held-out seasons 2022-2025, by game phase", x=0.01, ha="left", fontsize=11)
    fig.savefig(FIG / "wp_calibration.png", dpi=150)
    plt.close(fig)


def fig_teams(team: pd.DataFrame) -> None:
    t = team.sort_values("wp_lost_per_season")
    fig, ax = plt.subplots(figsize=(7, 8), constrained_layout=True)
    ax.barh(t.posteam, t.wp_lost_per_season, color=COL["go"], height=0.7)
    ax.set_xlabel("Win probability lost on clear-call 4th downs, per season (wins)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title("4th-down WP left on the table by team, 2014-2025", loc="left", fontsize=11)
    fig.savefig(FIG / "team_wp_lost.png", dpi=150)
    plt.close(fig)


def examples(R: pd.DataFrame) -> pd.DataFrame:
    # Examples come from outside the final two minutes of each half, where the
    # WP model has the most data and the fixed play clock matters least.
    clear = R[(R.confidence >= CLEAR_CONF) & (R.gap >= CLEAR_GAP) & (R.half_seconds_remaining > 120)]
    picks = {
        "Costliest punt": clear[(clear.decision == "punt") & (clear.best == "go")].nlargest(1, "wp_lost"),
        "Costliest field goal": clear[(clear.decision == "fg") & (clear.best == "go")].nlargest(1, "wp_lost"),
        "Costliest go-for-it": clear[(clear.decision == "go") & (clear.best != "go")].nlargest(1, "wp_lost"),
    }
    post = R[(R.season_type == "POST")].copy()
    post["stakes"] = post[["wp_go", "wp_fg", "wp_punt"]].max(axis=1) - post[["wp_go", "wp_fg", "wp_punt"]].min(axis=1)
    picks["Postseason coin flip"] = post[post.confidence.between(0.4, 0.6)].nlargest(1, "stakes")
    rows = []
    for k, r in picks.items():
        r = r.iloc[0]
        rows.append({"example": k, "game_date": r.game_date, "game_id": r.game_id, "posteam": r.posteam,
                     "defteam": r.defteam, "season_type": r.season_type, "score_diff": int(r.score_diff),
                     "clock_left_s": int(r.game_seconds_remaining), "ydstogo": int(r.ydstogo),
                     "yardline_100": int(r.yardline_100), "decision": r.decision, "best": r.best,
                     "wp_go": r.wp_go, "wp_fg": r.wp_fg, "wp_punt": r.wp_punt, "wp_lost": r.wp_lost,
                     "confidence": r.confidence, "desc": r.desc})
    return pd.DataFrame(rows)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    D = pd.read_parquet(RES / "decisions.parquet")
    R = relevant(D)
    clear = R[(R.confidence >= CLEAR_CONF) & (R.gap >= CLEAR_GAP)]
    st = season_table(R)
    st.to_csv(RES / "season_trend.csv", index=False)
    n_seasons = R.season.nunique()
    team = (clear.groupby("posteam").agg(wp_lost=("wp_lost", "sum"), n=("wp_lost", "size"))
            .assign(wp_lost_per_season=lambda d: d.wp_lost / n_seasons).reset_index())
    team.sort_values("wp_lost", ascending=False).to_csv(RES / "team_wp_lost.csv", index=False)
    coach = (clear.groupby("coach").agg(wp_lost=("wp_lost", "sum"), decisions=("wp_lost", "size"),
                                        seasons=("season", "nunique"), followed_bot=("wp_lost", lambda x: (x < 1e-12).mean()))
             .assign(wp_lost_per_season=lambda d: d.wp_lost / d.seasons).reset_index())
    coach = coach[coach.seasons >= 3].sort_values("wp_lost_per_season", ascending=False)
    coach.to_csv(RES / "coach_wp_lost.csv", index=False)
    ex = examples(R)
    ex.to_csv(RES / "examples.csv", index=False)

    by_action = {a: {"bot_says": int((clear.best == a).sum()),
                     "coach_agrees": float((clear[clear.best == a].decision == a).mean())} for a in ("go", "fg", "punt")}
    early = st[st.season <= 2017]
    late = st[st.season >= 2022]
    summary = {
        "fourth_downs_all": int(len(D)), "fourth_downs_relevant": int(len(R)),
        "clear_calls": int(len(clear)), "clear_share": float(len(clear) / len(R)),
        "agree_rate_all": float((R.best == R.decision).mean()),
        "agree_rate_clear": float((clear.best == clear.decision).mean()),
        "clear_by_action": by_action,
        "wp_lost_total": float(R.wp_lost.sum()), "wp_lost_clear": float(clear.wp_lost.sum()),
        "wp_lost_per_team_season_clear": float(clear.wp_lost.sum() / (32 * n_seasons)),
        "wp_lost_share_from_not_going": float(clear[clear.best == "go"].wp_lost.sum() / clear.wp_lost.sum()),
        "went_when_clear_go_2014_2017": float(np.average(early.went_when_clear_go, weights=early.clear_go_situations)),
        "went_when_clear_go_2022_2025": float(np.average(late.went_when_clear_go, weights=late.clear_go_situations)),
        "median_gap_clear": float(clear.gap.median()),
        "coin_flip_share": float((R.confidence < 0.6).mean()),
    }
    (RES / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(st.round(3).to_string(index=False))
    print(team.sort_values("wp_lost", ascending=False).head(5).round(2).to_string(index=False))
    print(coach.head(8).round(2).to_string(index=False))
    pd.set_option("display.max_colwidth", 220)
    print(ex.round(3).T.to_string())

    fig_decision_chart(pd.read_csv(RES / "decision_chart.csv"))
    fig_trend(st)
    fig_calibration(pd.read_csv(RES / "wp_calibration.csv"))
    fig_teams(team)
    print("figures written")


if __name__ == "__main__":
    main()

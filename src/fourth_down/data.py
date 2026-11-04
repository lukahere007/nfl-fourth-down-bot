"""Turn raw nflverse play-by-play into the slim tables the bot needs.

Everything here uses *pre-snap* information only for features. Labels
(win/loss, conversion, make/miss, next field position) are outcomes of the
play or game and are kept in clearly named ``label``/``outcome`` columns.

Tables produced by :func:`build_tables`:

``wp``          every scrimmage snap in regulation (training data for the WP model)
``conversion``  3rd/4th-down run/pass plays with the conversion outcome
``field_goal``  field-goal attempts with make/miss and kick conditions
``punt``        punts with the receiving team's next starting yardline
``gain``        yards gained beyond the line to gain on successful conversions
``fourth``      every 4th-down snap with the coach's actual decision
``kickoff``     median post-kickoff starting yardline by season
``xp``          extra-point success rate by season
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLS = [
    "game_id", "play_id", "season", "week", "season_type", "game_date", "home_team", "away_team",
    "posteam", "defteam", "qtr", "down", "ydstogo", "yardline_100", "game_seconds_remaining",
    "half_seconds_remaining", "score_differential", "posteam_timeouts_remaining",
    "defteam_timeouts_remaining", "spread_line", "result", "home_opening_kickoff", "play_type",
    "third_down_converted", "fourth_down_converted", "touchdown", "yards_gained",
    "field_goal_result", "kick_distance", "extra_point_result", "roof", "wind", "punt_blocked",
    "home_coach", "away_coach", "wp", "vegas_wp", "desc",
]

STATE_COLS = [
    "score_diff", "game_seconds_remaining", "half_seconds_remaining", "down", "ydstogo",
    "yardline_100", "pos_timeouts", "def_timeouts", "home", "spread", "receive_2h",
]


def load_raw(raw_dir: Path, seasons: list[int]) -> pd.DataFrame:
    """Read the needed columns for ``seasons`` from ``raw_dir``."""
    frames = []
    for s in seasons:
        path = Path(raw_dir) / f"play_by_play_{s}.parquet"
        frames.append(pd.read_parquet(path, columns=RAW_COLS))
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["game_id", "play_id"]).reset_index(drop=True)


def add_state(df: pd.DataFrame) -> pd.DataFrame:
    """Add offense-perspective pre-snap state columns (see ``STATE_COLS``)."""
    out = df.copy()
    is_home = (out["posteam"] == out["home_team"]).astype(int)
    out["home"] = is_home
    out["score_diff"] = out["score_differential"]
    out["pos_timeouts"] = out["posteam_timeouts_remaining"]
    out["def_timeouts"] = out["defteam_timeouts_remaining"]
    # spread_line is the home team's expected margin; convert to offense view.
    out["spread"] = np.where(is_home == 1, out["spread_line"], -out["spread_line"])
    # home_opening_kickoff == 1 means the home team kicked first, so it receives the 2nd half.
    home_gets_2h = out["home_opening_kickoff"] == 1
    pos_gets_2h = np.where(is_home == 1, home_gets_2h, ~home_gets_2h)
    out["receive_2h"] = np.where(out["qtr"] <= 2, pos_gets_2h, 0).astype(int)
    # Offense win label from the final result (home_score - away_score). Ties -> 0.5.
    res = np.where(is_home == 1, out["result"], -out["result"])
    out["label_win"] = np.select([res > 0, res < 0], [1.0, 0.0], 0.5)
    out["coach"] = np.where(is_home == 1, out["home_coach"], out["away_coach"])
    return out


def _snaps(df: pd.DataFrame) -> pd.DataFrame:
    """Regulation scrimmage snaps with a complete state."""
    m = (
        df["down"].between(1, 4)
        & df["qtr"].between(1, 4)
        & df["posteam"].notna()
        & df["play_type"].isin(["run", "pass", "punt", "field_goal", "qb_kneel", "qb_spike", "no_play"])
    )
    need = ["score_diff", "game_seconds_remaining", "half_seconds_remaining", "ydstogo",
            "yardline_100", "pos_timeouts", "def_timeouts", "spread"]
    return df[m].dropna(subset=need)


def decision_of(play_type: pd.Series) -> pd.Series:
    """Map nflverse ``play_type`` to the coach's 4th-down decision."""
    return play_type.map({"run": "go", "pass": "go", "field_goal": "fg", "punt": "punt"})


def _next_offense_start(df: pd.DataFrame) -> pd.DataFrame:
    """For each play, the posteam / yardline / quarter of the *next* snap in the same game.

    Non-snap rows (kickoffs, extra points, timeouts) are skipped, so after a punt
    or kickoff this is where the receiving offense actually started.
    """
    snap = df["down"].notna()
    tmp = df[["posteam", "yardline_100", "qtr"]].where(snap)
    shifted = tmp.groupby(df["game_id"]).shift(-1)
    filled = shifted.groupby(df["game_id"]).bfill()
    filled.columns = ["next_posteam", "next_yardline_100", "next_qtr"]
    return filled


def build_tables(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build every derived table from raw play-by-play (see module docstring)."""
    df = add_state(raw)
    nxt = _next_offense_start(df)
    df = pd.concat([df, nxt], axis=1)
    snaps = _snaps(df)

    keep = ["game_id", "play_id", "season", "week", "season_type", *STATE_COLS, "label_win", "wp", "vegas_wp"]
    wp = snaps[keep].copy()

    scrim = snaps[snaps["play_type"].isin(["run", "pass"]) & snaps["down"].isin([3, 4]) & (snaps["ydstogo"] <= 25)]
    conv = scrim[["game_id", "season", "down", "ydstogo", "yardline_100", "spread"]].copy()
    conv["converted"] = np.where(scrim["down"] == 3, scrim["third_down_converted"], scrim["fourth_down_converted"]).astype(int)

    fgp = snaps[snaps["play_type"] == "field_goal"].dropna(subset=["kick_distance"])
    fg = fgp[["game_id", "season", "kick_distance", "yardline_100"]].copy()
    fg["indoor"] = fgp["roof"].isin(["dome", "closed"]).astype(int)
    fg["wind"] = np.where(fg["indoor"] == 1, 0.0, fgp["wind"].fillna(fgp["wind"].median()))
    fg["made"] = (fgp["field_goal_result"] == "made").astype(int)

    pp = snaps[(snaps["play_type"] == "punt")]
    same_half = ((pp["qtr"] <= 2) == (pp["next_qtr"] <= 2))
    ok = (pp["next_posteam"] == pp["defteam"]) & same_half & pp["next_yardline_100"].notna()
    punt = pp.loc[ok, ["game_id", "season", "yardline_100"]].copy()
    punt["opp_yardline_100"] = pp.loc[ok, "next_yardline_100"].astype(float)
    punt["blocked"] = pp.loc[ok, "punt_blocked"].fillna(0).astype(int)

    succ = scrim[(conv["converted"] == 1) & (scrim["ydstogo"] < scrim["yardline_100"])]
    gain = succ[["game_id", "season", "ydstogo"]].copy()
    gain["extra"] = (succ["yards_gained"] - succ["ydstogo"]).clip(lower=0)

    f4 = snaps[snaps["down"] == 4].copy()
    f4["decision"] = decision_of(f4["play_type"])
    # Botched snaps on kicks show up as runs ("Aborted"); they were not decisions to go.
    aborted = f4["desc"].str.contains("Aborted", case=False, na=False) & (f4["decision"] == "go")
    f4 = f4[f4["decision"].notna() & ~aborted]
    fourth = f4[["game_id", "play_id", "season", "week", "season_type", "game_date", "posteam", "defteam",
                 "coach", *STATE_COLS, "roof", "wind", "decision", "label_win", "wp", "vegas_wp", "desc"]].copy()
    fourth["indoor"] = fourth["roof"].isin(["dome", "closed"]).astype(int)
    fourth["wind"] = np.where(fourth["indoor"] == 1, 0.0, fourth["wind"].fillna(fg["wind"].median()))
    fourth = fourth.drop(columns="roof")

    ko = df[(df["play_type"] == "kickoff")]
    ko_ok = (ko["next_posteam"] == ko["posteam"]) & ko["next_yardline_100"].notna()
    kickoff = (ko[ko_ok].groupby("season")["next_yardline_100"].median()
               .rename("kickoff_yardline_100").reset_index())

    xpp = df[df["play_type"] == "extra_point"]
    xp = (xpp.assign(good=(xpp["extra_point_result"] == "good").astype(int))
          .groupby("season")["good"].mean().rename("xp_rate").reset_index())

    return {"wp": wp, "conversion": conv, "field_goal": fg, "punt": punt, "gain": gain,
            "fourth": fourth, "kickoff": kickoff, "xp": xp}


def shrink(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns so committed parquet files stay small."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].astype("float32")
        elif pd.api.types.is_integer_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], downcast="integer")
    return out

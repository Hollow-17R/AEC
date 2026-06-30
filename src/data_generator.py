"""
Data Generator for E-Sports Player Performance Classification.

Generates ~2,000 realistic CS:GO professional player records with three
distinct player archetypes: Star, Support, and Entry/Average players.
Each archetype has correlated stat distributions that mimic real pro-level data.
"""

import itertools
import random
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.utils import (
    RAW_DATA_PATH,
    RANDOM_STATE,
    get_logger,
    save_dataframe,
    setup_logging,
)

logger = get_logger(__name__)

# ============================================================================
# Constants for data generation
# ============================================================================
TOTAL_PLAYERS = 2000
SEED = RANDOM_STATE  # 42

# Archetype proportions
STAR_RATIO = 0.20
SUPPORT_RATIO = 0.40
ENTRY_RATIO = 0.40

# Real CS:GO organisation names
TEAMS: List[str] = [
    "Navi", "FaZe", "Astralis", "G2", "Vitality", "Cloud9", "Heroic",
    "ENCE", "Liquid", "NIP", "MOUZ", "BIG", "Outsiders", "Fnatic",
    "FURIA", "Imperial", "paiN", "MIBR", "Spirit", "Virtus.pro",
    "Complexity", "OG", "Monte", "GamerLegion", "Eternal Fire",
    "TheMongolz", "Lynn Vision", "TYLOO", "Apeks", "SAW",
]

# Country pools weighted by CS:GO scene representation
COUNTRIES: List[str] = [
    "Denmark", "Sweden", "Finland", "Norway",         # Scandinavia
    "Russia", "Ukraine", "Poland", "Czech Republic",  # Eastern Europe
    "Brazil", "Argentina",                             # South America
    "France", "Germany", "Netherlands", "Belgium",     # Western Europe
    "Turkey", "Mongolia", "China", "Kazakhstan",       # Asia
    "United States", "Canada",                         # North America
    "Australia", "Bosnia", "Latvia", "Estonia",        # Other
]

# Player-name building blocks (pro-scene–style nicknames)
_PREFIXES = [
    "x", "z", "k", "d", "n", "sh", "bl", "cr", "st", "fr", "tr",
    "pr", "dr", "sw", "sn", "fl", "gl", "gr", "br", "cl", "sk",
]
_CORES = [
    "ace", "aze", "olt", "rix", "ush", "ink", "ong", "yre", "ell",
    "ant", "ops", "ven", "iko", "uno", "elt", "arn", "isk", "odo",
    "lex", "ent", "ark", "ion", "ade", "ume", "erg", "ost", "ulf",
]
_SUFFIXES = [
    "", "1", "x", "z", "0", "oo", "ie", "er", "zy", "ix", "us",
    "a", "o", "y", "i", "3r", "7", "9", "4k", "5s",
]
_STYLE_NAMES = [
    "NiKo", "s1mple", "ZywOo", "dev1ce", "coldzera", "ropz", "blameF",
    "rain", "karrigan", "gla1ve", "broky", "Twistzz", "EliGE", "NAF",
    "huNter", "nexa", "YEKINDAR", "Boombl4", "electroNic", "Perfecto",
    "stavn", "TeSeS", "cadiaN", "Magisk", "dupreeh", "KSCERATO",
    "yuurih", "arT", "drop", "FalleN", "fer", "TACO", "boltz",
    "shox", "apEX", "misutaaa", "Zontix", "headtr1ck", "iM", "Ax1Le",
    "sh1ro", "nafany", "HObbit", "Jame", "SANJI", "b1t", "m0NESY",
    "Snappi", "dycha", "hades", "Spinx", "flameZ", "tabseN", "syrsoN",
]


def _generate_unique_names(n: int, rng: np.random.Generator) -> List[str]:
    """Generate *n* unique player nicknames.

    Combines a pool of real-style names with procedurally generated ones
    to reach the required count.
    """
    names: set = set()

    # Seed with style names (shuffled)
    style_shuffled = list(_STYLE_NAMES)
    rng.shuffle(style_shuffled)
    for name in style_shuffled:
        if len(names) >= n:
            break
        names.add(name)

    # Generate remaining names procedurally
    combos = list(itertools.product(_PREFIXES, _CORES, _SUFFIXES))
    rng.shuffle(combos)
    for prefix, core, suffix in combos:
        if len(names) >= n:
            break
        candidate = prefix + core + suffix
        # Randomly capitalise first letter ~50 % of the time
        if rng.random() < 0.5:
            candidate = candidate.capitalize()
        names.add(candidate)

    # Fallback if we still don't have enough (very unlikely)
    idx = 0
    while len(names) < n:
        names.add(f"player_{idx}")
        idx += 1

    return list(names)[:n]


# ============================================================================
# Archetype stat generators
# ============================================================================

def _generate_star_stats(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate stats for Star players (~20 %).

    High kills, high rating (1.1–1.4), high headshot %, high damage.
    """
    maps_played = rng.integers(100, 500, size=n)
    rounds_per_map = rng.normal(25, 2, size=n).clip(20, 30).astype(int)
    rounds_played = maps_played * rounds_per_map

    rating = rng.normal(1.25, 0.08, size=n).clip(1.10, 1.45)
    kills_per_round = rng.normal(0.82, 0.06, size=n).clip(0.65, 1.00)
    deaths_per_round = rng.normal(0.58, 0.05, size=n).clip(0.42, 0.72)
    assists_per_round = rng.normal(0.10, 0.03, size=n).clip(0.03, 0.20)
    damage_per_round = rng.normal(88, 7, size=n).clip(70, 115)
    headshot_percentage = rng.normal(55, 6, size=n).clip(38, 72)

    total_kills = (kills_per_round * rounds_played).astype(int)
    total_deaths = (deaths_per_round * rounds_played).astype(int)
    total_assists = (assists_per_round * rounds_played).astype(int)

    opening_kills = (rng.normal(0.14, 0.03, size=n).clip(0.08, 0.22) * maps_played).astype(int)
    opening_deaths = (rng.normal(0.08, 0.02, size=n).clip(0.03, 0.14) * maps_played).astype(int)
    clutch_wins = (rng.normal(0.10, 0.03, size=n).clip(0.03, 0.20) * maps_played).astype(int)

    return pd.DataFrame({
        "maps_played": maps_played,
        "rounds_played": rounds_played,
        "total_kills": total_kills,
        "total_deaths": total_deaths,
        "total_assists": total_assists,
        "headshot_percentage": np.round(headshot_percentage, 1),
        "kills_per_round": np.round(kills_per_round, 2),
        "deaths_per_round": np.round(deaths_per_round, 2),
        "assists_per_round": np.round(assists_per_round, 2),
        "damage_per_round": np.round(damage_per_round, 1),
        "rating": np.round(rating, 2),
        "opening_kills": opening_kills,
        "opening_deaths": opening_deaths,
        "clutch_wins": clutch_wins,
    })


def _generate_support_stats(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate stats for Support players (~40 %).

    Moderate kills, moderate rating (0.9–1.1), higher assists.
    """
    maps_played = rng.integers(80, 450, size=n)
    rounds_per_map = rng.normal(25, 2, size=n).clip(20, 30).astype(int)
    rounds_played = maps_played * rounds_per_map

    rating = rng.normal(1.00, 0.06, size=n).clip(0.88, 1.12)
    kills_per_round = rng.normal(0.65, 0.05, size=n).clip(0.50, 0.80)
    deaths_per_round = rng.normal(0.64, 0.04, size=n).clip(0.50, 0.78)
    assists_per_round = rng.normal(0.16, 0.03, size=n).clip(0.08, 0.26)
    damage_per_round = rng.normal(72, 6, size=n).clip(55, 92)
    headshot_percentage = rng.normal(46, 5, size=n).clip(32, 62)

    total_kills = (kills_per_round * rounds_played).astype(int)
    total_deaths = (deaths_per_round * rounds_played).astype(int)
    total_assists = (assists_per_round * rounds_played).astype(int)

    opening_kills = (rng.normal(0.08, 0.02, size=n).clip(0.03, 0.14) * maps_played).astype(int)
    opening_deaths = (rng.normal(0.09, 0.02, size=n).clip(0.04, 0.15) * maps_played).astype(int)
    clutch_wins = (rng.normal(0.06, 0.02, size=n).clip(0.01, 0.12) * maps_played).astype(int)

    return pd.DataFrame({
        "maps_played": maps_played,
        "rounds_played": rounds_played,
        "total_kills": total_kills,
        "total_deaths": total_deaths,
        "total_assists": total_assists,
        "headshot_percentage": np.round(headshot_percentage, 1),
        "kills_per_round": np.round(kills_per_round, 2),
        "deaths_per_round": np.round(deaths_per_round, 2),
        "assists_per_round": np.round(assists_per_round, 2),
        "damage_per_round": np.round(damage_per_round, 1),
        "rating": np.round(rating, 2),
        "opening_kills": opening_kills,
        "opening_deaths": opening_deaths,
        "clutch_wins": clutch_wins,
    })


def _generate_entry_stats(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate stats for Entry / Average players (~40 %).

    Lower rating (0.7–1.0), higher deaths, aggressive opening duels.
    """
    maps_played = rng.integers(60, 400, size=n)
    rounds_per_map = rng.normal(25, 2, size=n).clip(20, 30).astype(int)
    rounds_played = maps_played * rounds_per_map

    rating = rng.normal(0.87, 0.07, size=n).clip(0.70, 1.02)
    kills_per_round = rng.normal(0.58, 0.06, size=n).clip(0.40, 0.75)
    deaths_per_round = rng.normal(0.70, 0.04, size=n).clip(0.58, 0.85)
    assists_per_round = rng.normal(0.12, 0.03, size=n).clip(0.04, 0.22)
    damage_per_round = rng.normal(65, 6, size=n).clip(48, 85)
    headshot_percentage = rng.normal(42, 6, size=n).clip(28, 58)

    total_kills = (kills_per_round * rounds_played).astype(int)
    total_deaths = (deaths_per_round * rounds_played).astype(int)
    total_assists = (assists_per_round * rounds_played).astype(int)

    # Entry fraggers: higher opening duel volume
    opening_kills = (rng.normal(0.12, 0.03, size=n).clip(0.05, 0.20) * maps_played).astype(int)
    opening_deaths = (rng.normal(0.13, 0.03, size=n).clip(0.06, 0.22) * maps_played).astype(int)
    clutch_wins = (rng.normal(0.04, 0.02, size=n).clip(0.00, 0.10) * maps_played).astype(int)

    return pd.DataFrame({
        "maps_played": maps_played,
        "rounds_played": rounds_played,
        "total_kills": total_kills,
        "total_deaths": total_deaths,
        "total_assists": total_assists,
        "headshot_percentage": np.round(headshot_percentage, 1),
        "kills_per_round": np.round(kills_per_round, 2),
        "deaths_per_round": np.round(deaths_per_round, 2),
        "assists_per_round": np.round(assists_per_round, 2),
        "damage_per_round": np.round(damage_per_round, 1),
        "rating": np.round(rating, 2),
        "opening_kills": opening_kills,
        "opening_deaths": opening_deaths,
        "clutch_wins": clutch_wins,
    })


# ============================================================================
# Public API
# ============================================================================

def generate_dataset() -> pd.DataFrame:
    """Generate a synthetic CS:GO player-stats dataset and save to disk.

    Returns
    -------
    pd.DataFrame
        DataFrame with ~2,000 rows and all required columns.
    """
    logger.info("Starting synthetic CS:GO dataset generation (seed=%d)", SEED)
    rng = np.random.default_rng(SEED)
    random.seed(SEED)

    n_star = int(TOTAL_PLAYERS * STAR_RATIO)
    n_support = int(TOTAL_PLAYERS * SUPPORT_RATIO)
    n_entry = TOTAL_PLAYERS - n_star - n_support  # remainder

    logger.info(
        "Archetype counts — Star: %d, Support: %d, Entry/Average: %d",
        n_star, n_support, n_entry,
    )

    # Generate per-archetype stats
    df_star = _generate_star_stats(n_star, rng)
    df_support = _generate_support_stats(n_support, rng)
    df_entry = _generate_entry_stats(n_entry, rng)

    df = pd.concat([df_star, df_support, df_entry], ignore_index=True)

    # Assign player names, teams, and countries
    names = _generate_unique_names(len(df), rng)
    rng.shuffle(names)
    df.insert(0, "player_name", names[: len(df)])
    df.insert(1, "team", rng.choice(TEAMS, size=len(df)))
    df.insert(2, "country", rng.choice(COUNTRIES, size=len(df)))

    # Shuffle rows so archetypes are interleaved
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    logger.info("Dataset generated: %d rows × %d columns", *df.shape)
    logger.info("Columns: %s", list(df.columns))

    # Persist
    save_dataframe(df, RAW_DATA_PATH)
    logger.info("Raw dataset saved → %s", RAW_DATA_PATH)

    return df


# ============================================================================
# CLI entry-point
# ============================================================================

if __name__ == "__main__":
    setup_logging()
    df = generate_dataset()
    print(f"\nGenerated {len(df)} player records.")
    print(df.head(10).to_string(index=False))
    print(f"\nSaved to {RAW_DATA_PATH}")

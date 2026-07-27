"""Pure lookup-table agent -- no engine search calls at all.

The search_agent.py approach (search_begin/search_step rollouts) kept
failing real submissions with "Validation Episode failed". Root cause,
confirmed from kaggle-environments' own PTCG spec: actTimeout=0 and a
shared remainingOverageTime=600s pool for the WHOLE episode -- there is no
free per-move allowance, so any repeated real engine simulation eventually
drains the clock across enough decisions/games.

This agent never calls search_begin/search_step. Every decision is O(1):
read the fields already present on the current Observation/Option (attack
damage via a cached all_attack() lookup, target Pokemon HP via the option's
area/index/playerIndex) and pick with static rules. Same computational
shape as the original 760-Elo baseline (effectively instant), but with
smarter per-selection-type rules than a flat priority order.
"""
from __future__ import annotations

import os
import random
import sys

from cg.api import (
    Observation, SelectType, SelectContext, OptionType, AreaType,
    to_observation_class, all_attack,
)

DEBUG = os.environ.get("PTCG_DEBUG", "1") == "1"

_ATTACK_DAMAGE = None


def _log(msg: str) -> None:
    if DEBUG:
        print(f"[ptcg-heuristic] {msg}", file=sys.stderr, flush=True)


def _attack_damage_lookup() -> dict:
    global _ATTACK_DAMAGE
    if _ATTACK_DAMAGE is None:
        _ATTACK_DAMAGE = {a.attackId: a.damage for a in all_attack()}
    return _ATTACK_DAMAGE


def read_deck_csv() -> list[int]:
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    with open(file_path, "r") as file:
        csv = file.read().split("\n")
    return [int(csv[i]) for i in range(60)]


def _pokemon_at(state, area: AreaType | None, index: int | None, player_index: int | None):
    """Resolve an Option's (area, index, playerIndex) to the actual Pokemon
    dataclass on the board, if it points at one. Returns None if not
    resolvable (e.g. area is DECK/DISCARD, or fields are missing)."""
    if area is None or index is None or player_index is None:
        return None
    try:
        ps = state.players[player_index]
    except (IndexError, TypeError):
        return None
    if area == AreaType.ACTIVE:
        return ps.active[0] if ps.active else None
    if area == AreaType.BENCH:
        return ps.bench[index] if 0 <= index < len(ps.bench) else None
    return None


def _hp_fraction(p) -> float:
    if p is None or not p.maxHp:
        return 0.0
    return max(0.0, p.hp) / p.maxHp


def _choose_main(obs: Observation) -> list[int]:
    """Develop before attacking: a MAIN select recurs multiple times per
    turn (attach/evolve/play are all separate steps you can take before
    ending your turn), and attacking normally ends the turn. Jumping
    straight to ATTACK the instant it's legal -- as an early version of
    this did -- skips every other beneficial action available that same
    turn (new energy attachments, evolutions, bench plays), which starves
    the board over time even though each individual attack looks fine in
    isolation. So: exhaust development options first, attack only once
    nothing else useful remains for this turn."""
    sel = obs.select
    options = sel.option
    n = len(options)

    priority = {
        OptionType.EVOLVE: 0, OptionType.ATTACH: 1, OptionType.PLAY: 2,
        OptionType.ABILITY: 3, OptionType.ATTACK: 4, OptionType.RETREAT: 5,
        OptionType.DISCARD: 6, OptionType.END: 7,
    }
    best_priority = min(priority.get(o.type, 6) for o in options)
    tied = [i for i, o in enumerate(options) if priority.get(o.type, 6) == best_priority]

    if options[tied[0]].type == OptionType.ATTACK and len(tied) > 1:
        attack_dmg = _attack_damage_lookup()
        best = max(tied, key=lambda i: attack_dmg.get(options[i].attackId, 0))
        _log(f"MAIN: attacking (highest damage {attack_dmg.get(options[best].attackId, 0)} "
             f"among {len(tied)} attack options), no better development action available")
        return [best]

    best = tied[0]
    _log(f"MAIN: development priority rule chose {options[best].type} "
         f"({len(tied)} tied at priority {best_priority})")
    return [best]


def _choose_card(obs: Observation) -> list[int]:
    """CARD selections: for switch/to-active contexts, prefer the
    healthiest resolvable Pokemon. For discard/to-deck-style contexts,
    prefer discarding the WEAKEST resolvable Pokemon (keep the strong
    ones). Falls back to random for anything else / unresolvable options."""
    sel = obs.select
    options = sel.option
    state = obs.current
    n = len(options)
    idx = list(range(n))

    prefer_high_hp = {
        SelectContext.SWITCH, SelectContext.TO_ACTIVE, SelectContext.TO_BENCH,
        SelectContext.TO_FIELD, SelectContext.HEAL, SelectContext.REMOVE_DAMAGE_COUNTER,
    }
    prefer_low_hp = {SelectContext.DISCARD, SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM}

    if sel.context in prefer_high_hp or sel.context in prefer_low_hp:
        resolved = [
            (i, _pokemon_at(state, o.area, o.index, o.playerIndex))
            for i, o in enumerate(options)
        ]
        resolvable = [(i, p) for i, p in resolved if p is not None]
        if resolvable:
            if sel.context in prefer_high_hp:
                best_i, _ = max(resolvable, key=lambda ip: _hp_fraction(ip[1]))
            else:
                best_i, _ = min(resolvable, key=lambda ip: _hp_fraction(ip[1]))
            return [best_i]

    k = max(sel.minCount, min(sel.maxCount, n))
    return random.sample(idx, k) if n and k else []


def _greedy_choice(obs: Observation) -> list[int]:
    sel = obs.select
    options = sel.option
    n = len(options)
    idx = list(range(n))

    if sel.type == SelectType.MAIN and sel.maxCount >= 1:
        return _choose_main(obs)

    if sel.type in (SelectType.CARD, SelectType.CARD_OR_ATTACHED_CARD):
        return _choose_card(obs)

    if sel.type == SelectType.YES_NO:
        if sel.context == SelectContext.MULLIGAN:
            return [0] if sel.maxCount >= 1 else []
        return random.sample(idx, sel.maxCount) if n else []

    return random.sample(idx, sel.maxCount) if n else []


def agent(obs_dict: dict) -> list[int]:
    try:
        obs: Observation = to_observation_class(obs_dict)
        if obs.select is None:
            return read_deck_csv()

        sel = obs.select
        _log(f"decision: type={sel.type.name if hasattr(sel.type,'name') else sel.type} "
             f"context={sel.context.name if hasattr(sel.context,'name') else sel.context} "
             f"n_options={len(sel.option)}")
        return _greedy_choice(obs)
    except Exception:
        sel = obs_dict.get("select")
        if not sel:
            return []
        n = len(sel.get("option", []))
        k = max(sel.get("minCount", 0), min(sel.get("maxCount", 0), n))
        return random.sample(range(n), k) if n and k else []

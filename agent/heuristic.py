"""Heuristic evaluation of a PTCG battle State, from one player's perspective.

Used both as (a) a fallback scorer for non-MAIN selections and (b) the leaf
evaluation function for the 1-ply search over MAIN options in search_agent.py.
"""
from __future__ import annotations

from cg.api import State, PlayerState, Pokemon


def _pokemon_value(p: Pokemon | None) -> float:
    if p is None:
        return 0.0
    hp_frac = max(0.0, p.hp) / max(1, p.maxHp)
    value = 2.0 + 3.0 * hp_frac  # being in play at all is worth something; HP fraction refines it
    value += 0.4 * len(p.energyCards)  # energy investment = board development
    value += 0.3 * len(p.tools)
    return value


def _player_value(ps: PlayerState) -> float:
    value = 0.0
    active = ps.active[0] if ps.active else None
    value += _pokemon_value(active) * 1.5  # active matters most (it's what's attacking/being attacked)
    for benched in ps.bench:
        value += _pokemon_value(benched)
    value += 0.15 * ps.handCount  # cards in hand = future options
    value += 0.05 * ps.deckCount  # avoid decking out
    if active is not None:
        if ps.poisoned:
            value -= 0.5
        if ps.burned:
            value -= 0.5
        if ps.asleep:
            value -= 0.8
        if ps.paralyzed:
            value -= 0.8
        if ps.confused:
            value -= 0.4
    # Prizes remaining: fewer remaining prizes = closer to winning, so this
    # is scored as a POSITIVE for having taken more (see evaluate_state).
    return value


def evaluate_state(state: State, your_index: int) -> float:
    """Higher is better for `your_index`. Roughly zero-sum plus a small
    constant-sum board-development term (both players' board states count
    positively, so this isn't purely differential -- but the differential
    dominates for move comparison since it's evaluated from the same root)."""
    if state.result == your_index:
        return 1e6
    if state.result == 1 - your_index:
        return -1e6
    if state.result == 2:  # draw
        return 0.0

    opp_index = 1 - your_index
    you = state.players[your_index]
    opp = state.players[opp_index]

    score = _player_value(you) - _player_value(opp)

    # Prizes: a player takes ONE OF THEIR OWN prize cards each time they
    # score a knockout. So `you.prize` shrinking means YOU got a knockout
    # (good); `opp.prize` shrinking means the OPPONENT got a knockout on you
    # (bad). Reward low you.prize / high opp.prize.
    score += 8.0 * (len(opp.prize) - len(you.prize))

    return score

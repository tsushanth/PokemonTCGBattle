"""Local test harness: run full battles between search_agent and a random
baseline using the real compiled engine, entirely offline (no Kaggle needed).
"""
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cg.game import battle_start, battle_select, battle_finish
from cg.api import to_observation_class

import search_agent


def random_agent(obs_dict: dict, deck: list[int]) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return deck
    n = len(obs.select.option)
    return random.sample(list(range(n)), obs.select.maxCount)


def load_deck(path: str) -> list[int]:
    with open(path) as f:
        lines = [l.strip() for l in f.read().split("\n") if l.strip()]
    return [int(x) for x in lines[:60]]


def run_battle(deck0: list[int], deck1: list[int], agent0, agent1, max_selects=4000, verbose=False):
    obs_dict, start_data = battle_start(deck0, deck1)
    if obs_dict is None:
        return None, f"battle_start failed: errorPlayer={start_data.errorPlayer} errorType={start_data.errorType}"

    n = 0
    while n < max_selects:
        n += 1
        obs = to_observation_class(obs_dict)
        if obs.current is not None and obs.current.result != -1:
            result = obs.current.result
            battle_finish()
            return result, f"finished after {n} selects"

        if obs.select is None:
            # initial deck submission step: engine wants each player's deck
            # via the observation loop too, in some engine builds -- handle
            # defensively even though battle_start already took decks.
            choice = deck0
        else:
            yi = obs.current.yourIndex if obs.current else 0
            agent_fn, deck = (agent0, deck0) if yi == 0 else (agent1, deck1)
            choice = agent_fn(obs_dict, deck) if agent_fn is random_agent else agent_fn(obs_dict)
        if verbose:
            sel = obs.select
            print(f"  select#{n}: type={sel.type if sel else None} choice={choice}")
        obs_dict = battle_select(choice)

    battle_finish()
    return None, f"hit max_selects={max_selects} without finishing"


if __name__ == "__main__":
    N_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    deck = load_deck(os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv"))

    def search_agent_fn(obs_dict):
        return search_agent.agent(obs_dict)

    def rand_agent_fn(obs_dict):
        return random_agent(obs_dict, deck)

    wins = {0: 0, 1: 0, None: 0}
    t0 = time.time()
    for g in range(N_GAMES):
        # Alternate which side is search_agent to control for first-player advantage.
        if g % 2 == 0:
            agents = (search_agent_fn, rand_agent_fn)
            search_side = 0
        else:
            agents = (rand_agent_fn, search_agent_fn)
            search_side = 1
        result, msg = run_battle(deck, deck, agents[0], agents[1])
        tag = "search_agent" if result == search_side else ("random" if result in (0, 1) else "draw/timeout")
        print(f"game {g}: result={result} winner={tag} ({msg})")
        wins[result] = wins.get(result, 0) + 1

    print(f"\ndone in {time.time()-t0:.1f}s")
    print("raw results:", wins)

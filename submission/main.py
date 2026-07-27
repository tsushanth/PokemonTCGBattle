from heuristic_agent import agent  # noqa: F401 -- Kaggle harness imports `agent` from this module

# NOTE: no sys.path / __file__ manipulation here. The official sample
# submission's main.py does a plain `from cg.api import ...` with zero path
# hacks, and this project's own history recorded fixing a __file__
# NameError from an earlier version -- the exec()-based harness has no
# __file__ in its globals. heuristic_agent.py and search_agent.py also both
# do plain `from cg.api import ...` for the same reason.

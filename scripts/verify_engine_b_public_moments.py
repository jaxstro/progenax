"""Replay the retained full reverse reference through the public moment API."""

from pathlib import Path

import diagnose_engine_b_uniform_lookup as replay
from diagnose_engine_b_population_moments import model_at


def public_moment(theta, key):
    return model_at(theta).estimate_population_moments(
        key, 100000, 1.0, n_realizations=4, draws_per_component=50000
    )["mean_speed_squared"]


if __name__ == "__main__":
    replay.prototype.moment = public_moment
    replay.OUT = Path(
        "docs/reviews/plummer-om-artifacts/engine-b-public-moment-replay.json"
    )
    replay.main()

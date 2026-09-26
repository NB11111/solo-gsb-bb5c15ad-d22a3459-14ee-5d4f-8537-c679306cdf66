"""Bounded extreme-value sampling for the timeline channels.

``services.downsample`` keeps the first/last samples of the queried range,
splits the middle samples into time-ordered buckets and — within the
remaining point budget — keeps each bucket's minimum and maximum, so an
isolated spike survives sampling and the result never exceeds max_points.
``GET /api/timeline`` uses it per channel while count/shown and the fault,
maintenance, suppressed and annotation layers keep using the original data.
"""

import json
import os
import tempfile

import pytest

from app import db, services
from app.engine import to_epoch


def at(minute: int) -> float:
    return to_epoch(f"2026-09-23T00:{minute:02d}:00Z")


@pytest.fixture()
def scratch_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db.DB_PATH = tmp.name
    db._conn = None
    db.init_db()
    yield tmp.name
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(tmp.name + suffix)
        except OSError:
            pass


def series(points):
    """[(minute, value), ...] -> (ts, value, event_id) rows sorted by ts."""
    return [(at(m), float(v), i + 1) for i, (m, v) in enumerate(points)]


# --------------------------------------------------------------------------- #
# services.downsample
# --------------------------------------------------------------------------- #


def test_under_budget_returns_everything():
    values = series([(0, 1), (1, 5), (2, 2), (3, 4)])
    assert services.downsample(values, 4) == values
    assert services.downsample(values, 10) == values
    assert services.downsample([], 5) == []


def test_never_exceeds_budget_and_keeps_first_last():
    values = series([(m, (m * 7) % 5) for m in range(20)])
    for max_points in range(2, 10):
        out = services.downsample(values, max_points)
        assert len(out) <= max_points
        assert out[0] == values[0] and out[-1] == values[-1]
        # original (ts, event id) order
        assert [(p[0], p[2]) for p in out] == sorted((p[0], p[2]) for p in out)


def test_stride_bug_fixed_twenty_samples_four_points():
    # The old stride sampler returned 5 points here and dropped the spike.
    values = series([(m, 100 if m == 7 else 10) for m in range(20)])
    out = services.downsample(values, 4)
    assert len(out) <= 4
    assert out[0] == values[0] and out[-1] == values[-1]
    assert any(p[1] == 100.0 and p[0] == at(7) for p in out)


def test_isolated_spike_and_dip_survive():
    values = series([(m, 10) for m in range(30)])
    values[10] = (values[10][0], 99.0, values[10][2])   # spike
    values[20] = (values[20][0], -50.0, values[20][2])  # dip
    out = services.downsample(values, 6)
    kept = {p[1] for p in out}
    assert 99.0 in kept and -50.0 in kept


def test_bucket_min_and_max_both_kept():
    # 8 samples, budget 6 -> two 3-sample buckets, each gives min+max.
    values = series([(0, 0), (1, 3), (2, -2), (3, 1), (4, 4), (5, -1), (6, 2), (7, 0)])
    out = services.downsample(values, 6)
    assert len(out) == 6
    assert [p[1] for p in out] == [0.0, 3.0, -2.0, 4.0, -1.0, 0.0]


def test_constant_bucket_contributes_once():
    values = series([(m, 5) for m in range(10)])
    out = services.downsample(values, 4)
    assert len(out) <= 4
    assert out[0] == values[0] and out[-1] == values[-1]


def test_single_slot_keeps_larger_baseline_deviation():
    # max_points=3 -> one middle bucket with a single slot.  The bucket's
    # first->last baseline slopes from (min1, 0) to (min9, 10); the min
    # candidate (min8, -1) deviates more from it than the max (min4, 11),
    # even though 11 is the larger absolute value.
    values = series([(0, 0), (1, 0), (4, 11), (8, -1), (9, 10), (10, 10)])
    out = services.downsample(values, 3)
    assert [p[0] for p in out] == [at(0), at(8), at(10)]


def test_single_slot_tie_keeps_earlier_sample():
    # Flat baseline 0: min -1 (min2) and max +1 (min3) deviate equally;
    # the earlier sample wins.
    values = series([(0, 0), (1, 0), (2, -1), (3, 1), (4, 0), (5, 0)])
    out = services.downsample(values, 3)
    assert [p[0] for p in out] == [at(0), at(2), at(5)]


def test_odd_budget_leaves_single_slot_for_last_bucket():
    # max_points=5 -> budget 3 -> two buckets; the last one gets one slot
    # and keeps its larger-deviation extreme (the min9 spike).
    values = series(
        [(0, 0), (1, 0.5), (2, -0.5), (3, 0.4), (4, -0.4), (5, 0), (6, 9), (7, -1), (8, 0), (9, 0)]
    )
    out = services.downsample(values, 5)
    assert len(out) == 5
    assert any(p[0] == at(6) and p[1] == 9.0 for p in out)


def test_tiny_budgets():
    values = series([(m, m % 3) for m in range(10)])
    assert services.downsample(values, 2) == [values[0], values[-1]]
    assert services.downsample(values, 1) == [values[0]]
    assert services.downsample(values, 0) == []


# --------------------------------------------------------------------------- #
# GET /api/timeline
# --------------------------------------------------------------------------- #


@pytest.fixture()
def client(scratch_db):
    from fastapi.testclient import TestClient
    from app.main import app

    return TestClient(app)


def _import_spike_series():
    rows = [
        {
            "device": "pump-1",
            "metric": "temp",
            "ts": f"2026-09-23T00:{m:02d}:00Z",
            "value": 100 if m == 7 else 10,
        }
        for m in range(20)
    ]
    return services.import_events("\n".join(json.dumps(r) for r in rows))


def test_api_timeline_bounded_sampling_keeps_spike(client):
    _import_spike_series()
    body = client.get("/api/timeline", params={"max_points": 4}).json()
    (ch,) = body["channels"]
    # count/shown still describe the original data
    assert ch["count"] == 20 and ch["shown"] == 20
    values = ch["values"]
    assert len(values) <= 4
    assert values[0]["ts"] == at(0) and values[-1]["ts"] == at(19)
    assert any(p["v"] == 100.0 and p["ts"] == at(7) for p in values)
    assert [p["ts"] for p in values] == sorted(p["ts"] for p in values)


def test_api_timeline_under_budget_returns_all_samples(client):
    _import_spike_series()
    body = client.get("/api/timeline", params={"max_points": 400}).json()
    (ch,) = body["channels"]
    assert len(ch["values"]) == 20


def test_api_timeline_other_layers_use_original_data(client):
    _import_spike_series()
    rule = client.post(
        "/api/rules",
        json={
            "name": "高温",
            "logic": "AND",
            "duration": 0,
            "conditions": [
                {"device": "pump-1", "metric": "temp", "op": ">=", "threshold": 80}
            ],
        },
    ).json()
    client.post("/api/evaluate")
    client.post(
        "/api/maintenance",
        json={
            "device": "pump-1",
            "name": "检修",
            "start": "2026-09-23T00:02:00Z",
            "end": "2026-09-23T00:03:00Z",
        },
    )
    body = client.get("/api/timeline", params={"max_points": 4}).json()
    # fault layer: the minute-7 spike still confirms a fault [00:07, 00:08)
    faults = [(f["rule_id"], f["start"], f["end"]) for f in body["faults"]]
    assert (rule["id"], at(7), at(8)) in faults
    # maintenance layer untouched by channel sampling
    assert [(w["name"], w["start"], w["end"]) for w in body["maintenance"]] == [
        ("检修", at(2), at(3))
    ]

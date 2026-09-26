"""Bounded extreme-value sampling for the timeline channels.

``services.downsample`` keeps the first/last samples of the queried range,
splits the middle samples into time-ordered buckets and — within the
remaining point budget — keeps each bucket's minimum and maximum, so an
isolated spike survives sampling and the result never exceeds max_points.
``GET /api/timeline`` uses it per channel while count/shown and the fault,
maintenance, suppressed and annotation layers keep using the original data.

The timeline read path no longer fetchall()s a channel's range: inside one
read-only snapshot it first counts the range samples, then scans the rows
in bounded fetchmany batches straight into fixed-size per-bucket
accumulators, so a channel's extra work memory is bounded by max_points
(plus one scan batch) rather than the raw row count.  The streaming path
produces exactly ``downsample(fetchall(...))`` output; see the equivalence,
large-sample and concurrent-write tests below.
"""

import json
import os
import random
import tempfile
import threading
import time

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


# --------------------------------------------------------------------------- #
# Streaming read path: bounded-memory batch scan in one read snapshot
# --------------------------------------------------------------------------- #


def _numeric_series(values):
    """Insert (ts = minute index, value) samples for channel d/m."""
    rows = [
        {"device": "d", "metric": "m", "ts": m, "value": v}
        for m, v in enumerate(values)
    ]
    services.import_events("\n".join(json.dumps(r) for r in rows), reevaluate=False)


def _fetchall_reference(start, end, max_points):
    """downsample() over every in-range row materialized at once."""
    clause, params = services._channel_range_clause(start, end)
    rows = db.query(
        f"SELECT ts, value, id FROM events{clause} ORDER BY ts, id",
        ("d", "m", *params),
    )
    all_values = [(r["ts"], r["value"], r["id"]) for r in rows]
    return services.downsample(all_values, max_points), len(all_values)


@pytest.mark.parametrize("max_points", [2, 3, 4, 5, 6, 7, 8, 25, 400])
@pytest.mark.parametrize("n", [0, 1, 2, 3, 7, 8, 9, 100, 1001])
def test_streaming_path_equals_fetchall_downsample(scratch_db, n, max_points):
    rng = random.Random(n * 31 + max_points)
    # Mostly-tied values exercise the constant-bucket / first-occurrence ties.
    values = [rng.choice([-5, -1, 0, 0, 0, 1, 7]) for _ in range(n)]
    _numeric_series(values)
    start = 1 if n > 2 else None
    end = n - 2 if n > 4 else None
    with db.read_snapshot():
        got, total = services._downsample_channel_samples("d", "m", start, end, max_points)
    expected, exp_total = _fetchall_reference(start, end, max_points)
    assert (got, total) == (expected, exp_total)
    assert len(got) <= max_points


def test_streaming_scan_uses_bounded_batches(scratch_db, monkeypatch):
    _numeric_series([i % 13 for i in range(5000)])
    batch_sizes = []
    original = db.query_batches

    def spy(sql, params=(), batch_size=1000):
        for batch in original(sql, params, batch_size):
            batch_sizes.append(len(batch))
            assert len(batch) <= batch_size
            yield batch

    monkeypatch.setattr(db, "query_batches", spy)
    with db.read_snapshot():
        got, total = services._downsample_channel_samples("d", "m", None, None, 400)
    assert total == 5000
    assert len(got) <= 400
    # The interior scan must have been paged, never materialized wholesale.
    assert batch_sizes
    assert max(batch_sizes) == services._TIMELINE_SCAN_BATCH == 1000


def test_large_series_count_exact_and_extremes_survive(scratch_db, monkeypatch):
    n = 50_000
    values = [5.0] * n
    values[12_345] = 999.0   # isolated spike
    values[40_000] = -999.0  # isolated dip
    _numeric_series(values)

    max_batch = 0
    original = db.query_batches

    def spy(sql, params=(), batch_size=1000):
        nonlocal max_batch
        for batch in original(sql, params, batch_size):
            max_batch = max(max_batch, len(batch))
            assert len(batch) <= batch_size
            yield batch

    monkeypatch.setattr(db, "query_batches", spy)
    body = services.timeline_data(None, None, 400)
    (channel,) = body["channels"]
    assert channel["count"] == n
    assert channel["shown"] == n
    points = {(p["ts"], p["v"]) for p in channel["values"]}
    assert (12_345.0, 999.0) in points
    assert (40_000.0, -999.0) in points
    assert channel["values"][0]["ts"] == 0.0
    assert channel["values"][-1]["ts"] == float(n - 1)
    assert len(channel["values"]) <= 400
    assert max_batch <= 1000


class _HeldWrite:
    """Keep an INSERT uncommitted (then roll back) on the writer connection."""

    def __init__(self, sql, params=(), rollback=True):
        self.sql, self.params, self.rollback = sql, params, rollback
        self.started = threading.Event()
        self.release = threading.Event()
        self.done = threading.Event()
        self.error = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            conn = db.get_conn()
            with db._write_lock:
                conn.execute(self.sql, tuple(self.params))
                self.started.set()
                self.release.wait(10)
                if self.rollback:
                    conn.rollback()
                else:
                    conn.commit()
        except Exception as exc:  # pragma: no cover - surfaced by the test
            self.error = exc
        finally:
            self.done.set()

    def __enter__(self):
        self._thread.start()
        assert self.started.wait(5)
        time.sleep(0.05)
        return self

    def __exit__(self, *exc):
        self.release.set()
        assert self.done.wait(5)
        assert self.error is None, self.error


def test_streaming_timeline_hides_uncommitted_in_range_rows(client):
    _numeric_series([float(i) for i in range(5000)])
    held = _HeldWrite(
        "INSERT INTO events (device, metric, ts, value, raw, import_id)"
        " VALUES ('d','m',?,9999.0,'x',NULL)",
        (900_000.0,),
    )
    with held:
        body = client.get("/api/timeline", params={"max_points": 400}).json()
    (channel,) = body["channels"]
    assert channel["shown"] == 5000
    assert channel["values"][-1]["ts"] == 4999.0
    assert body["data_end"] == 4999.0
    # Rolled back: still invisible after the writer exits.
    body = client.get("/api/timeline", params={"max_points": 400}).json()
    assert body["channels"][0]["shown"] == 5000


def test_commit_mid_scan_keeps_one_committed_view(scratch_db, monkeypatch):
    _numeric_series([5.0] * 10_000)
    barrier = threading.Event()
    results = []
    original = db.query_batches

    def gated(sql, params=(), batch_size=1000):
        for index, batch in enumerate(original(sql, params, batch_size)):
            # Once the interior bucket scan has delivered its second batch,
            # let another thread commit new in-range rows before the scan
            # continues.
            if index == 1 and "ts > ? AND ts < ?" in sql:
                barrier.set()
                time.sleep(0.3)
            yield batch

    monkeypatch.setattr(db, "query_batches", gated)

    def reader():
        with db.read_snapshot():
            results.append(
                services._downsample_channel_samples("d", "m", None, None, 400)
            )

    t = threading.Thread(target=reader)
    t.start()
    assert barrier.wait(5)
    with db.transaction() as tx:
        for k in range(1000):
            tx.execute(
                "INSERT INTO events (device, metric, ts, value, raw, import_id)"
                " VALUES ('d','m',?,7.0,'x',NULL)",
                (20_000 + k,),
            )
    t.join(10)
    assert not t.is_alive()

    values, total = results[0]
    # The pinned snapshot predates the commit: count, interior scan (the
    # service asserts it consumed total-2 rows) and endpoints stay coherent.
    assert total == 10_000
    assert values[0][0] == 0.0 and values[-1][0] == 9999.0
    assert all(ts < 10_000 for ts, _, _ in values)

    # The next request gets a fresh snapshot including the committed rows.
    body = services.timeline_data(None, None, 400)
    assert body["channels"][0]["shown"] == 11_000
    assert body["channels"][0]["values"][-1]["ts"] == 20_999.0


def test_streaming_snapshot_shares_timeline_layers(client):
    # count/shown and the fault/maintenance layers still come from the same
    # committed view as the batch-scanned channel points.
    _numeric_series([90.0 if i == 7 else 10.0 for i in range(3000)])
    rule = client.post(
        "/api/rules",
        json={
            "name": "高温",
            "logic": "AND",
            "duration": 0,
            "conditions": [
                {"device": "d", "metric": "m", "op": ">=", "threshold": 80}
            ],
        },
    ).json()
    client.post("/api/evaluate")
    client.post(
        "/api/maintenance",
        json={
            "device": "d",
            "name": "检修",
            "start": 100.0,
            "end": 200.0,
        },
    )
    body = client.get("/api/timeline", params={"max_points": 400}).json()
    (channel,) = body["channels"]
    assert channel["count"] == 3000 and channel["shown"] == 3000
    assert any(p["ts"] == 7.0 and p["v"] == 90.0 for p in channel["values"])
    assert (rule["id"], 7.0, 8.0) in {
        (f["rule_id"], f["start"], f["end"]) for f in body["faults"]
    }
    assert [(w["name"], w["start"], w["end"]) for w in body["maintenance"]] == [
        ("检修", 100.0, 200.0)
    ]

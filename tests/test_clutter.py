"""Sea-clutter study on synthetic labels and events with known shares (Phase 7 step 5; D-058)."""

import numpy as np
import pytest

from evflow.maritime import clutter

H, W = 10, 20


def _masks():
    lab = np.full((H, W), 1, np.uint8)        # water
    lab[:4] = 2                                # sky: 80 px
    lab[4, :10] = 0                            # obstacle: 10 px
    lab[-1, -2:] = 255                         # ignore: 2 px -> water 108 px
    from evflow.io.lars import class_masks
    return class_masks(lab)


def _events(n_water, n_sky, n_obs, n_ign=0):
    x = [5] * n_water + [3] * n_sky + [2] * n_obs + [19] * n_ign
    y = [7] * n_water + [1] * n_sky + [4] * n_obs + [9] * n_ign
    return np.array(x), np.array(y)


def test_window_counts_and_clip_stats():
    m = _masks()
    c = clutter.window_class_counts(*_events(60, 20, 10, 10), m)
    assert c["n_events"] == 100 and c["n_px"] == 200
    assert c["classes"]["water"] == {"events": 60, "px": 108}
    assert c["classes"]["ignore"] == {"events": 10, "px": 2}
    s = clutter.clip_stats([c])["classes"]
    assert s["water"]["event_share"] == pytest.approx(0.6)
    assert s["water"]["pixel_share"] == pytest.approx(108 / 200)
    assert s["water"]["enrichment"] == pytest.approx(0.6 / 0.54)
    assert s["sky"]["events_per_px"] == pytest.approx(20 / 80)
    # two windows of one clip: per-pixel rate stays per window
    s2 = clutter.clip_stats([c, c])["classes"]
    assert s2["sky"]["events_per_px"] == pytest.approx(20 / 80)
    assert s2["water"]["event_share"] == pytest.approx(0.6)
    with pytest.raises(ValueError):
        clutter.window_class_counts(np.array([W]), np.array([0]), m)


def test_pooled_is_event_weighted_with_ci():
    m = _masks()
    clips = [clutter.clip_stats([clutter.window_class_counts(*_events(a, b, 0), m)])
             for a, b in ((90, 10), (10, 90), (50, 50), (70, 30))]
    p = clutter.pooled(clips, resamples=500)
    assert p["water"]["event_share"]["mean"] == pytest.approx(220 / 400)
    lo, hi = p["water"]["event_share"]["ci95"]
    assert lo <= 220 / 400 <= hi and hi - lo > 0.05
    assert p["obstacle"]["event_share"]["mean"] == 0.0
    assert p["water"]["pixel_share"]["mean"] == pytest.approx(108 / 200)
    assert p["water"]["n_clips_with_class"] == 4


def test_attribute_breakdown_levels_and_spearman():
    m = _masks()
    rng = np.random.default_rng(0)
    clips, attrs = [], []
    waves = ["still"] * 4 + ["moderate"] * 6 + ["rough"] * 2
    for i, wv in enumerate(waves):
        nw = {"still": 20, "moderate": 50, "rough": 90}[wv] + int(rng.integers(0, 5))
        clips.append(clutter.clip_stats([clutter.window_class_counts(*_events(nw, 30, 5), m)]))
        attrs.append({"waves": wv, "reflections": None if i == 0 else "none",
                      "special.glitter": i % 2 == 0, "scene_type": "sea_like",
                      "lighting": "day_like"})
    out = clutter.attribute_breakdown(clips, attrs)
    lv = out["waves"]["levels"]
    assert lv["still"]["n"] == 4 and lv["rough"] == {"n": 2, "reported": False,
                                                     "reason": "fewer than 3 clips"}
    assert lv["moderate"]["water_enrichment"] > lv["still"]["water_enrichment"]
    rho = out["waves"]["spearman_water_enrichment"]
    assert rho["rho"] > 0.8 and rho["n"] == 12
    # reflections: one level only (+ null) -> no correlation
    assert out["reflections"]["spearman"]["reported"] is False
    assert out["reflections"]["levels"]["null"]["n"] == 1
    assert "spearman_water_enrichment" in out["special.glitter"]
    assert "spearman" not in out["lighting"]

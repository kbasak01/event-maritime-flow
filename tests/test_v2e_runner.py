"""sim/v2e_runner logic without running v2e: profiles, argv policy, warnings, clock map."""

from pathlib import Path

import numpy as np
import pytest

from evflow.sim import v2e_runner as r

DOC = {
    "v2e_commit_file": "configs/v2e_commit.txt",
    "common": ["--no_preview", "--skip_video_output", "--overwrite"],
    "profiles": {
        "trial": {"allow_dvs_params": True, "size": ["--dvs346"],
                  "args": {"--dvs_params": "clean"}},
        "grid": {"size": ["--dvs346"],
                 "args": {"--auto_timestamp_resolution": "True", "--pos_thres": "0.2",
                          "--neg_thres": "0.2", "--cutoff_hz": "60"}},
        "child": {"base": "grid", "args": {"--pos_thres": "0.3"}},
        "big": {"base": "child", "size": ["--output_width", "640", "--output_height", "360"]},
        "sneaky": {"base": "grid", "args": {"--dvs_params": "noisy"}},
        "runflag": {"base": "grid", "args": {"--dvs_h5": "x.h5"}},
    },
}
RUN = dict(input_path="/in", output_folder="/out", dvs_h5="u.h5", seed=7, input_frame_rate=43.75)


def test_resolve_inheritance():
    p = r.resolve_profile(DOC, "big")
    assert p["size"] == ["--output_width", "640", "--output_height", "360"]
    assert p["args"]["--pos_thres"] == "0.3" and p["args"]["--cutoff_hz"] == "60"


def test_override_only_existing_flags():
    p = r.resolve_profile(DOC, "grid", {"--pos_thres": 0.15, "--neg_thres": 0.15})
    assert p["args"]["--pos_thres"] == "0.15"
    with pytest.raises(r.PolicyError, match="not a flag"):
        r.resolve_profile(DOC, "grid", {"--pos_thresh": 0.1})


def test_argv_contents_and_order():
    argv = r.build_argv(DOC, "child", **RUN, overrides={"--cutoff_hz": 0}, stop_time=5)
    assert argv[:4] == ["--no_preview", "--skip_video_output", "--overwrite", "--dvs346"]
    kv = {argv[i]: argv[i + 1] for i in range(len(argv) - 1) if argv[i].startswith("--")}
    assert kv["--pos_thres"] == "0.3" and kv["--cutoff_hz"] == "0"
    assert kv["--dvs_emulator_seed"] == "7" and kv["--input_frame_rate"] == "43.75"
    assert kv["--stop_time"] == "5.0" and kv["--dvs_h5"] == "u.h5"
    assert Path(kv["--input"]).is_absolute() and Path(kv["--output_folder"]).is_absolute()


def test_dvs_params_only_in_trial():
    r.build_argv(DOC, "trial", **RUN)
    with pytest.raises(r.PolicyError, match="dvs_params"):
        r.build_argv(DOC, "sneaky", **RUN)


def test_profile_cannot_set_run_flags():
    with pytest.raises(r.PolicyError, match="runner-owned"):
        r.build_argv(DOC, "runflag", **RUN)


def test_policy_requires_frame_rate_and_seed():
    with pytest.raises(r.PolicyError, match="input_frame_rate"):
        r.check_policy(["--no_preview", "--skip_video_output", "--dvs_h5", "a",
                        "--dvs_emulator_seed", "1"])
    with pytest.raises(r.PolicyError, match="bad --input_frame_rate"):
        r.build_argv(DOC, "grid", **{**RUN, "input_frame_rate": 0.0})


def test_scan_log():
    log = ("INFO ok\nWARNING Lowpass 3dB cutoff ... \n but this results in large IIR mixing "
           "factor eps = dt/tau = 1.873 >  0.3 (maxeps),\nWARNING nonmonotonic timestamp(s) at "
           "indices [[3]]\n")
    s = r.scan_log(log)
    assert len(s["undersampling"]) == 1 and s["nonmonotonic"] == 1
    assert r.scan_log("all fine\n") == {"undersampling": [], "nonmonotonic": 0,
                                         "too_many_events": 0}
    w = "WARNING:v2ecore.emulator:Too many events generated for this frame: num_iter=304>100"
    assert r.scan_log(f"dvs: 3%|x[1;31m{w}\r{w}\n")["too_many_events"] == 2


def test_clock_map_knots_and_linear():
    fps = 40.0
    real = 1_000_000 + np.array([0, 25_010, 49_990, 75_000], np.int64)   # jittered 25 ms
    fn, st = r.make_clock_map(real, fps)
    uni = r.uniform_clock_us(4, fps)
    np.testing.assert_array_equal(fn(uni), real - real[0])               # knots exact
    assert fn(np.array([12_500.0]))[0] == 12_505                          # midpoint, linear
    assert st["uniform_vs_real_max_abs_dev_us"] == 10.0
    assert np.all(np.diff(fn(np.linspace(0, 75_000, 1000))) >= 0)          # monotone


def test_size_from_args():
    assert r._size_from_args(["--dvs346"]) == (346, 260)
    assert r._size_from_args(["--output_width", "640", "--output_height", "360"]) == (640, 360)
    with pytest.raises(r.PolicyError):
        r._size_from_args([])


def test_repo_profiles_resolve():
    doc = r.load_profiles()
    for name in doc["profiles"]:
        p = r.resolve_profile(doc, name)
        assert p["size"], name
    assert "--dvs_params" not in r.resolve_profile(doc, "mvsec_grid")["args"]


def test_export_video_frames_range(tmp_path):
    import cv2

    vid = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"MJPG"), 30, (32, 24))
    if not w.isOpened():
        pytest.skip("no MJPG writer in this OpenCV build")
    for k in range(10):
        w.write(np.full((24, 32, 3), k * 20, np.uint8))
    w.release()
    ex = r.export_video_frames(vid, tmp_path / "out", 2, 5)
    files = sorted((tmp_path / "out" / "frames").glob("*.png"))
    assert ex["n_frames"] == 4 and len(files) == 4
    first = cv2.imread(str(files[0]))
    assert first.shape == (24, 32, 3) and abs(int(first.mean()) - 40) <= 3   # frame 2
    with pytest.raises(ValueError):
        r.export_video_frames(vid, tmp_path / "bad", 8, 12)


def test_run_unit_fails_on_clamped_events(tmp_path, monkeypatch):
    import evflow.io.v2e_reader as rd

    def fake_run(argv, *, v2e_python, v2e_repo, log_path):
        folder = Path(argv[argv.index("--output_folder") + 1])
        (folder / argv[argv.index("--dvs_h5") + 1]).write_bytes(b"x")
        log_path.write_text("ok\n")
        return {"returncode": 0, "wall_time_s": 1.0, "gpu_mem_used_peak_mib": None}

    monkeypatch.setattr(r, "run_v2e", fake_run)
    monkeypatch.setattr(r, "gpu_name", lambda: "test")
    monkeypatch.setattr(rd, "convert_v2e_h5", lambda *a, **k: {
        "n_events": 1000, "n_events_beyond_clock_map": 5})
    doc = {**DOC, "v2e_commit_file": "configs/v2e_commit.txt"}
    with pytest.raises(r.V2EFailed, match="clamped"):
        r.run_unit(unit="u", out_dir=tmp_path, profile="grid", seed=1, input_path=tmp_path,
                   input_sha256="0" * 64, input_desc={}, input_frame_rate=40.0,
                   frame_rate_source="test", width=346, height=260, source="s", sensor="s",
                   clock_real_ts_us=np.arange(5, dtype=np.int64) * 25_000,
                   profiles_doc=doc, paths={"v2e": {"python": "py", "repo": "repo"}})


def test_git_dirty_sees_untracked_code(tmp_path, monkeypatch):
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.setattr(r, "REPO", tmp_path)
    assert r.git_dirty() is False
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "new.py").write_text("x = 1\n")
    assert r.git_dirty() is True

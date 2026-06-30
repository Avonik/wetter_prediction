from wetter import config


def test_core_constants():
    assert config.STATION_ID == "06093"
    assert config.STATION_ELEV_M == 62.0
    assert config.MODELS == ["icon_d2", "icon_eu", "icon_global", "gfs_seamless", "ecmwf_ifs025"]
    assert config.LEAD_TIMES_H == [24, 48, 72, 96, 120, 144, 168]


def test_raw_path_creates_parents(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    p = config.raw_path("obs", "x.parquet")
    assert p == tmp_path / "raw" / "obs" / "x.parquet"
    assert p.parent.is_dir()

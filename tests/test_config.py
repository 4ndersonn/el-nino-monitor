"""Testes da configuração local."""

from pathlib import Path

from el_nino_monitor.config import Nino34Region, Settings, get_settings


def test_default_paths_are_under_project_root() -> None:
    settings = Settings()
    assert settings.raw_dir == settings.project_root / "data" / "raw"
    assert settings.processed_dir == settings.project_root / "data" / "processed"
    assert settings.maps_dir == settings.project_root / "outputs" / "maps"
    assert settings.reports_dir == settings.project_root / "outputs" / "reports"
    assert settings.nino34 == Nino34Region(-5.0, 5.0, 190.0, 240.0)


def test_environment_overrides_data_dir_and_log_level(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EL_NINO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EL_NINO_LOG_LEVEL", "DEBUG")

    settings = get_settings()

    assert settings.data_dir == tmp_path.resolve()
    assert settings.log_level == "DEBUG"


def test_create_directories(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", output_dir=tmp_path / "outputs")
    settings.create_directories()
    assert all(
        directory.is_dir()
        for directory in (
            settings.raw_dir,
            settings.processed_dir,
            settings.maps_dir,
            settings.reports_dir,
        )
    )

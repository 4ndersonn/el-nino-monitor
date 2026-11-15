"""Parâmetros e caminhos compartilhados da aplicação."""

from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLIMATOLOGY_START_YEAR = 1991
CLIMATOLOGY_END_YEAR = 2020


@dataclass(frozen=True)
class Nino34Region:
    """Limites Niño 3.4 em graus: longitude leste de 0 a 360."""

    latitude_min: float = -5.0
    latitude_max: float = 5.0
    longitude_min: float = 190.0
    longitude_max: float = 240.0


@dataclass(frozen=True)
class Settings:
    """Configuração local da aplicação."""

    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    output_dir: Path = PROJECT_ROOT / "outputs"
    log_level: str = "INFO"
    nino34: Nino34Region = Nino34Region()

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def maps_dir(self) -> Path:
        return self.output_dir / "maps"

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / "reports"

    def create_directories(self) -> None:
        """Cria somente os diretórios necessários para arquivos gerados."""
        for directory in (
            self.raw_dir,
            self.processed_dir,
            self.maps_dir,
            self.reports_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Lê as opções locais do ambiente no momento da chamada."""
    data_dir = os.getenv("EL_NINO_DATA_DIR")
    log_level = os.getenv("EL_NINO_LOG_LEVEL", "INFO")
    return Settings(
        data_dir=Path(data_dir).expanduser().resolve() if data_dir else PROJECT_ROOT / "data",
        log_level=log_level,
    )

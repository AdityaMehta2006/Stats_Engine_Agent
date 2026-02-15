"""
Application settings loaded from environment variables.
Uses Pydantic BaseSettings for validation and type safety.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    """Central configuration - all values loaded from .env file."""

    # LLM
    gemini_api_key: str = Field(..., description="Google Gemini API key")

    # MySQL
    mysql_host: str = Field(default="localhost")
    mysql_port: int = Field(default=3306)
    mysql_user: str = Field(default="root")
    mysql_password: str = Field(..., description="MySQL root password")

    # Upload
    max_upload_size_mb: int = Field(default=50)

    # Paths (derived, not from .env)
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @property
    def uploads_dir(self) -> Path:
        path = self.base_dir / "uploads"
        path.mkdir(exist_ok=True)
        return path

    @property
    def outputs_dir(self) -> Path:
        path = self.base_dir / "outputs"
        path.mkdir(exist_ok=True)
        return path

    def run_dir(self, csv_stem: str) -> Path:
        """Per-CSV run folder: outputs/{csv_stem}/"""
        safe = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in csv_stem)
        path = self.outputs_dir / safe
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run_charts_dir(self, csv_stem: str) -> Path:
        """Per-CSV charts folder: outputs/{csv_stem}/charts/"""
        path = self.run_dir(csv_stem) / "charts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run_report_path(self, csv_stem: str) -> Path:
        """Per-CSV report path: outputs/{csv_stem}/report.md"""
        return self.run_dir(csv_stem) / "report.md"

    @property
    def mysql_url(self) -> str:
        """Connection URL without a specific database (for creating new DBs)."""
        return f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}@{self.mysql_host}:{self.mysql_port}"

    def mysql_db_url(self, db_name: str) -> str:
        """Connection URL targeting a specific database."""
        return f"{self.mysql_url}/{db_name}"


# Singleton instance
_settings = None


def get_settings() -> Settings:
    """Returns a cached Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

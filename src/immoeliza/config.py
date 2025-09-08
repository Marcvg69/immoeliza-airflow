from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    data_root: str = "./data"
    analytics_db: str = "./analytics/immoeliza.duckdb"
    models_dir: str = "./models"
    scrape_user_agent: str = "Mozilla/5.0 (compatible; ImmoElizaBot/0.1)"
    model_config = SettingsConfigDict(env_prefix="IMMO_", env_file=".env", extra="ignore")

settings = Settings()

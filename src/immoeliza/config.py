from pydantic import BaseSettings

class Settings(BaseSettings):
    data_root: str = "./data"
    analytics_db: str = "./analytics/immoeliza.duckdb"
    models_dir: str = "./models"
    scrape_user_agent: str = "Mozilla/5.0 (compatible; ImmoElizaBot/0.1)"

    class Config:
        env_prefix = "IMMO_"
        env_file = ".env"

settings = Settings()

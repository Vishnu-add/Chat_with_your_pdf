import os
from chromadb.config import Settings

# Define Chroma Settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl = 'duckdb+parquet' ,
    persist_directory = "db",
    anonymized_telemetry = False
)
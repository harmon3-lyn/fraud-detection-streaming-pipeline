from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
TRAIN_CSV  = BASE_DIR / "fraudTrain.csv"
TEST_CSV   = BASE_DIR / "fraudTest.csv"
OUTPUT_DIR = BASE_DIR / "data" / "synthetic"
ZIP_PATH   = BASE_DIR / "data" / "synthetic_data.zip"

RANDOM_SEED = 42

# SQL Server
SQL_SERVER   = "localhost\\SQLEXPRESS"
SQL_DATABASE = "FraudDetection"
SQL_DRIVER   = "ODBC Driver 18 for SQL Server"
SQL_TRUST_CERT = True

USE_WINDOWS_AUTH = True  

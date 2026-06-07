# config.py — Fixed Central configuration for the Uploaded IBM Telco Dataset
from pathlib import Path

BASE_DIR        = Path(__file__).resolve().parent
DATA_RAW        = BASE_DIR / "data" / "raw"
DATA_PROCESSED  = BASE_DIR / "data" / "processed"
MODELS_DIR      = BASE_DIR / "src" / "models"
REPORTS_DIR     = BASE_DIR / "reports"

# Match the exact column names inside your downloaded CSV file
TARGET_COL      = "Churn Value"
ID_COL          = "CustomerID"

NUMERIC_COLS = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
]

CATEGORICAL_COLS = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
]

TEST_SIZE       = 0.20
RANDOM_STATE    = 42
CV_FOLDS        = 5
CLASSIFICATION_THRESHOLD = 0.40

AVG_CUSTOMER_LTV        = 1200   
RETENTION_CAMPAIGN_COST = 50     
RETENTION_SUCCESS_RATE  = 0.30
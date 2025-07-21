import json
import secrets
from pathlib import Path
import pandas as pd

# === Set your actual base directory here ===
BASE_DIR = Path("/Developer/serverdata/nebula-clients")
OUTPUT_TABLE = BASE_DIR / "token_table.csv"

# Track tokens for summary table
token_records = []

# Ensure base directory exists
if not BASE_DIR.exists():
    print(f"[ERROR] Base directory not found: {BASE_DIR}")
    exit(1)

# Process each guestXX / jetsonXX folder
for folder in sorted(BASE_DIR.iterdir()):
    if folder.is_dir() and (folder.name.startswith("guest") or folder.name.startswith("jetson")):
        # Generate a secure random token
        token = secrets.token_hex(16)
        tokens = [token]

        # Write tokens.json
        token_path = folder / "tokens.json"
        with open(token_path, "w") as f:
            json.dump(tokens, f, indent=2)

        print(f"[INFO] Wrote token for {folder.name}: {token}")
        token_records.append({
            "client": folder.name,
            "token": token
        })

# Write CSV summary
df = pd.DataFrame(token_records)
df.to_csv(OUTPUT_TABLE, index=False)
print(f"[✅] Token table written to: {OUTPUT_TABLE}")
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import os
import subprocess
import re
from fastapi import APIRouter
from fastapi import Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
#pip install fastapi uvicorn

bearer_scheme = HTTPBearer()
VALID_TOKENS = ["supersecrettoken123", "another-token"]

def verify_token(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)):
    token = credentials.credentials
    if token not in VALID_TOKENS:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing token",
        )

app = FastAPI()

#BASE_DIR = Path(__file__).parent
# ✅ Use an absolute path for client files
BASE_DIR = Path("/opt/sjsujetson/nebula/clients").resolve()

CLIENTS_DIR = BASE_DIR / "clients"

BINARIES = {
    "macos-arm64": BASE_DIR / "binaries/nebula-darwin-arm64",
    "linux-amd64": BASE_DIR / "binaries/nebula-linux-amd64",
    "linux-arm64": BASE_DIR / "binaries/nebula-linux-arm64"
}

@app.get("/nebula/{client}/{filename}")
def get_file(client: str, filename: str):
    file_path = BASE_DIR / client / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

# ✅ Nebula status API
@app.get("/nebula/status")
def nebula_status(auth: HTTPAuthorizationCredentials = Security(verify_token)):
    status = {
        "nebula_running": False,
        "tun_device": None,
        "nebula_ip": None,
        "lighthouse_peer": None,
    }

    # Check if nebula process is running
    try:
        result = subprocess.run(["pgrep", "-f", "nebula"], stdout=subprocess.PIPE)
        status["nebula_running"] = bool(result.stdout.strip())
    except Exception:
        pass

    # Check if nebula1 exists and get IP
    try:
        ip_info = subprocess.check_output(["ip", "addr", "show", "nebula1"]).decode()
        status["tun_device"] = "nebula1"
        ip_match = re.search(r"inet (\d+\.\d+\.\d+\.\d+)/", ip_info)
        if ip_match:
            status["nebula_ip"] = ip_match.group(1)
    except Exception:
        status["tun_device"] = "not found"

    # Optionally: parse `ip route` or `config.yml` to extract lighthouse
    try:
        with open("/etc/nebula/config.yml") as f:
            for line in f:
                if "static_host_map" in line or "hosts:" in line:
                    status["lighthouse_peer"] = []
                if re.match(r'^\s*-\s*"', line):
                    peer = line.strip().strip("-").strip('"')
                    status["lighthouse_peer"].append(peer)
    except Exception:
        pass

    return JSONResponse(status)

# ✅ List all available clients
@app.get("/nebula/members")
def list_members():
    members = []
    if not BASE_DIR.exists():
        return []

    for client_dir in BASE_DIR.iterdir():
        if client_dir.is_dir():
            name = client_dir.name
            ip_guess = None
            try:
                num = int(re.search(r'\d+$', name).group())
                if name.startswith("jetson"):
                    ip_guess = f"192.168.100.{10 + num}"
                elif name.startswith("guest"):
                    ip_guess = f"192.168.100.{200 + num}"
            except:
                pass
            members.append({"name": name, "ip_guess": ip_guess})

    return JSONResponse(members)

@app.get("/nebula/peers")
def get_nebula_peers():
    try:
        result = subprocess.check_output(
            ["nebula", "-config", "/etc/nebula/config.yml", "-test"]
        ).decode()

        peers = []
        peer_blocks = result.strip().split("\n\n")
        for block in peer_blocks:
            peer = {}
            for line in block.splitlines():
                if "Remote VPN IP" in line:
                    peer["vpn_ip"] = line.split(":")[1].strip()
                if "Remote" in line and "VPN" not in line:
                    peer["public_endpoint"] = line.split(":")[1].strip()
                if "Latency" in line:
                    peer["latency"] = line.split(":")[1].strip()
            if peer:
                peers.append(peer)

        return {"online_peers": peers}
    except Exception as e:
        return {"error": str(e)}

#User can download zip file, unzip, and perform nebula run in a single step
# curl -O -G \
#   -d "token=mysecrettoken" \
#   -d "platform=macos-arm64" \
#   http://your-server:8000/download/guest01

# unzip guest01.zip
# ./run.sh
@app.get("/download/{guest_id}")
def download_bundle(guest_id: str, token: str = Query(...), platform: str = Query(...)):
    guest_dir = CLIENTS_DIR / guest_id
    if not guest_dir.exists():
        raise HTTPException(status_code=404, detail="Guest not found")

    # Check token file
    token_file = guest_dir / "tokens.json"
    if not token_file.exists():
        raise HTTPException(status_code=403, detail="Token file missing")

    import json
    allowed_tokens = json.loads(token_file.read_text())
    if token not in allowed_tokens:
        raise HTTPException(status_code=403, detail="Invalid token")

    if platform not in BINARIES:
        raise HTTPException(status_code=400, detail="Unsupported platform")

    # Create ZIP bundle
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        shutil.copy(guest_dir / f"{guest_id}.crt", temp_path / f"{guest_id}.crt")
        shutil.copy(guest_dir / f"{guest_id}.key", temp_path / f"{guest_id}.key")
        shutil.copy(guest_dir / "config.yml", temp_path / "config.yml")
        shutil.copy(BINARIES[platform], temp_path / "nebula")
        (temp_path / "nebula").chmod(0o755)

        # Create run.sh
        (temp_path / "run.sh").write_text("""#!/bin/bash
chmod +x nebula
./nebula -config config.yml
""")
        (temp_path / "run.sh").chmod(0o755)

        # ZIP it
        zip_path = temp_path / f"{guest_id}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file in temp_path.iterdir():
                zipf.write(file, arcname=file.name)

        return FileResponse(zip_path, filename=f"{guest_id}.zip")

#uvicorn main:app --host 0.0.0.0 --port 8000
#http://IP:8000/nebula/jetson01/config.yml
#run in the backend:
#nohup uvicorn main:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &

# curl -O -G \
#   -d "token=mysecrettoken" \
#   -d "platform=macos-arm64" \
#   http://your-server:8000/download/guest01

# unzip guest01.zip
# ./run.sh
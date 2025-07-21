#!/usr/bin/env python3
"""
Nebula VPN Management API Server

A FastAPI-based server for managing Nebula VPN clients, providing endpoints for:
- Client configuration file distribution
- Network status monitoring
- Peer discovery and management
- Secure client bundle downloads

Dependencies: pip install fastapi uvicorn
Usage: uvicorn main:app --host 0.0.0.0 --port 8000
"""

# Standard library imports
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union

# Third-party imports
from fastapi import FastAPI, HTTPException, Query, Security, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

# Security configuration
bearer_scheme = HTTPBearer()
VALID_TOKENS = ["jetsonsupertoken", "sjsujetsontoken"]  # TODO: Move to environment variables

# Configuration constants
CONFIG_FILE_PATH = "/etc/nebula/config.yml"
NEBULA_INTERFACE = "nebula1"
MAX_PEERS_DISPLAY = 50  # Limit peer display for performance

def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
    client_path: Optional[Path] = None
) -> None:
    """
    Unified token verification method that supports both global and client-specific tokens.
    
    Args:
        credentials: HTTP authorization credentials containing the bearer token
        client_path: Optional path to client directory for client-specific token verification
        
    Raises:
        HTTPException: If token is invalid or missing
    """
    token = credentials.credentials
    
    # If client_path is provided, verify against client-specific tokens
    if client_path is not None:
        _verify_client_specific_token(client_path, token)
    else:
        # Verify against global tokens for API endpoints
        if token not in VALID_TOKENS:
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing token",
                headers={"WWW-Authenticate": "Bearer"},
            )


def _verify_client_specific_token(client_path: Path, token: str) -> None:
    """
    Verify the provided token against client's allowed tokens.
    
    Args:
        client_path: Path to client directory
        token: Token to verify
        
    Raises:
        HTTPException: If token verification fails
    """
    token_file = client_path / "tokens.json"
    if not token_file.exists():
        raise HTTPException(status_code=403, detail="Client access tokens not configured")

    try:
        with open(token_file, 'r') as f:
            allowed_tokens = json.load(f)
            
        if not isinstance(allowed_tokens, list):
            raise HTTPException(status_code=500, detail="Invalid token configuration format")
            
        if token not in allowed_tokens:
            raise HTTPException(status_code=403, detail="Invalid or expired token")
            
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Corrupted token configuration")
    except (PermissionError, OSError) as e:
        raise HTTPException(status_code=500, detail=f"Cannot access token configuration: {str(e)}")

# FastAPI application setup
app = FastAPI(
    title="Nebula VPN Management API",
    description="API for managing Nebula VPN clients and network monitoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware for better performance and security
app.add_middleware(GZipMiddleware, minimum_size=1000)  # Compress responses > 1KB
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict to specific origins in production
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

# Directory configuration
# Using absolute path for client files to ensure consistency across deployments
BASE_DIR = Path("/Developer/serverdata").resolve()
CLIENTS_DIR = BASE_DIR / "nebula-clients"
PUBLIC_DIR = BASE_DIR / "public"  # Public directory for downloadable files

# Platform-specific binary mappings
# Maps platform identifiers to their corresponding Nebula binary paths
BINARIES: Dict[str, Path] = {
    "macos-arm64": BASE_DIR / "binaries/nebula-darwin-arm64",
    "linux-amd64": BASE_DIR / "binaries/nebula-linux-amd64",
    "linux-arm64": BASE_DIR / "binaries/nebula-linux-arm64"
}

# Validate directory structure on startup
if not BASE_DIR.exists():
    raise RuntimeError(f"Base directory does not exist: {BASE_DIR}")
if not CLIENTS_DIR.exists():
    CLIENTS_DIR.mkdir(parents=True, exist_ok=True)
if not PUBLIC_DIR.exists():
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/nebula/{client}/{filename}")
def get_file(client: str, filename: str) -> FileResponse:
    """
    Serve individual client configuration files.
    
    Args:
        client: Client identifier (e.g., 'jetson01', 'guest01')
        filename: Requested filename (e.g., 'config.yml', 'client.crt')
        
    Returns:
        FileResponse: The requested file
        
    Raises:
        HTTPException: 404 if file not found, 400 if path traversal detected
    """
    # Security: Prevent path traversal attacks
    if ".." in client or ".." in filename or "/" in client:
        raise HTTPException(status_code=400, detail="Invalid client or filename")
    
    file_path = CLIENTS_DIR / client / filename
    
    # Ensure the resolved path is still within CLIENTS_DIR
    try:
        file_path.resolve().relative_to(CLIENTS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file path")
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)

@app.get("/nebula/status")
def nebula_status(auth: HTTPAuthorizationCredentials = Security(verify_token)) -> JSONResponse:
    """
    Get comprehensive Nebula VPN status information.
    
    Returns network status including process state, tunnel device info,
    IP configuration, and lighthouse peer information.
    
    Args:
        auth: Bearer token authentication
        
    Returns:
        JSONResponse: Status information including:
            - nebula_running: Boolean indicating if Nebula process is active
            - tun_device: Name of tunnel device or "not found"
            - nebula_ip: IP address assigned to tunnel interface
            - lighthouse_peer: List of configured lighthouse peers
    """
    status = {
        "nebula_running": False,
        "tun_device": None,
        "nebula_ip": None,
        "lighthouse_peer": None,
        "process_count": 0,
    }

    # Check if nebula process is running with improved error handling
    try:
        result = subprocess.run(
            ["pgrep", "-f", "nebula"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            timeout=5  # Prevent hanging
        )
        if result.stdout.strip():
            status["nebula_running"] = True
            # Count number of nebula processes
            status["process_count"] = len(result.stdout.strip().split(b'\n'))
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        # pgrep might not be available on all systems
        status["nebula_running"] = False

    # Check tunnel device status and extract IP information
    try:
        ip_info = subprocess.check_output(
            ["ip", "addr", "show", NEBULA_INTERFACE],
            stderr=subprocess.DEVNULL,
            timeout=5
        ).decode()
        status["tun_device"] = NEBULA_INTERFACE
        
        # Extract IP address with improved regex
        ip_match = re.search(r"inet (\d+\.\d+\.\d+\.\d+)/(\d+)", ip_info)
        if ip_match:
            status["nebula_ip"] = ip_match.group(1)
            status["subnet_mask"] = ip_match.group(2)
            
        # Extract interface state
        if "state UP" in ip_info:
            status["interface_state"] = "UP"
        elif "state DOWN" in ip_info:
            status["interface_state"] = "DOWN"
            
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        status["tun_device"] = "not found"

    # Parse lighthouse configuration from config file
    status["lighthouse_peer"] = _parse_lighthouse_config()

    return JSONResponse(status)


def _parse_lighthouse_config() -> Optional[List[str]]:
    """
    Parse lighthouse peer configuration from Nebula config file.
    
    Returns:
        List of lighthouse peer addresses or None if parsing fails
    """
    try:
        with open(CONFIG_FILE_PATH, 'r') as f:
            content = f.read()
            
        # Look for static_host_map section
        lighthouse_peers = []
        in_static_host_map = False
        
        for line in content.splitlines():
            line = line.strip()
            
            if "static_host_map:" in line:
                in_static_host_map = True
                continue
                
            if in_static_host_map:
                # End of static_host_map section
                if line and not line.startswith(' ') and not line.startswith('-'):
                    break
                    
                # Extract peer addresses
                peer_match = re.match(r'^\s*["\']?([^"\':]+):["\']?\s*\[?["\']?([^"\']]+)["\']?', line)
                if peer_match:
                    lighthouse_peers.append(f"{peer_match.group(1)}:{peer_match.group(2)}")
                    
        return lighthouse_peers if lighthouse_peers else None
        
    except (FileNotFoundError, PermissionError, IOError):
        return None

@app.get("/debug/server-info")
def debug_server_info() -> JSONResponse:
    """
    Debug endpoint to check server configuration and directory structure.
    
    Returns:
        JSONResponse: Server configuration and directory information
    """
    info = {
        "base_dir": str(BASE_DIR),
        "base_dir_exists": BASE_DIR.exists(),
        "clients_dir": str(CLIENTS_DIR),
        "clients_dir_exists": CLIENTS_DIR.exists(),
        "binaries": {},
        "clients": []
    }
    
    # Check binary availability
    for platform, binary_path in BINARIES.items():
        info["binaries"][platform] = {
            "path": str(binary_path),
            "exists": binary_path.exists(),
            "readable": binary_path.exists() and os.access(binary_path, os.R_OK)
        }
    
    # List client directories if they exist
    if CLIENTS_DIR.exists():
        try:
            for client_dir in CLIENTS_DIR.iterdir():
                if client_dir.is_dir():
                    client_info = {
                        "name": client_dir.name,
                        "path": str(client_dir),
                        "files": {}
                    }
                    
                    # Check for required files
                    required_files = {
                        "tokens.json": client_dir / "tokens.json",
                        "config.yml": client_dir / "config.yml",
                        f"{client_dir.name}.crt": client_dir / f"{client_dir.name}.crt",
                        f"{client_dir.name}.key": client_dir / f"{client_dir.name}.key"
                    }
                    
                    for file_name, file_path in required_files.items():
                        client_info["files"][file_name] = {
                            "exists": file_path.exists(),
                            "readable": file_path.exists() and os.access(file_path, os.R_OK)
                        }
                    
                    info["clients"].append(client_info)
        except Exception as e:
            info["clients_error"] = str(e)
    
    return JSONResponse(info)

@app.get("/nebula/members")
def list_members() -> JSONResponse:
    """
    List all available Nebula VPN client configurations.
    
    Scans the clients directory and returns information about each client,
    including predicted IP addresses based on naming conventions.
    
    Returns:
        JSONResponse: List of client information including:
            - name: Client identifier
            - ip_guess: Predicted IP address based on naming convention
            - config_exists: Whether config.yml exists for this client
            - cert_exists: Whether certificate files exist
    """
    members = []
    
    # Early return if clients directory doesn't exist
    if not CLIENTS_DIR.exists():
        return JSONResponse([])

    try:
        for client_dir in CLIENTS_DIR.iterdir():
            if not client_dir.is_dir():
                continue
                
            name = client_dir.name
            
            # Skip hidden directories and system directories
            if name.startswith('.') or name in ['__pycache__', 'tmp']:
                continue
            
            # Predict IP address based on naming convention
            ip_guess = _predict_client_ip(name)
            
            # Check for required configuration files
            config_exists = (client_dir / "config.yml").exists()
            cert_exists = (
                (client_dir / f"{name}.crt").exists() and 
                (client_dir / f"{name}.key").exists()
            )
            
            member_info = {
                "name": name,
                "ip_guess": ip_guess,
                "config_exists": config_exists,
                "cert_exists": cert_exists,
                "ready": config_exists and cert_exists
            }
            
            members.append(member_info)
            
    except (PermissionError, OSError) as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error accessing client directory: {str(e)}"
        )

    # Sort members by name for consistent ordering
    members.sort(key=lambda x: x["name"])
    
    return JSONResponse(members)


def _predict_client_ip(client_name: str) -> Optional[str]:
    """
    Predict client IP address based on naming convention.
    
    Args:
        client_name: Name of the client (e.g., 'jetson01', 'guest05')
        
    Returns:
        Predicted IP address or None if pattern doesn't match
    """
    try:
        # Extract number from end of client name
        number_match = re.search(r'(\d+)$', client_name)
        if not number_match:
            return None
            
        num = int(number_match.group(1))
        
        # Apply IP prediction rules based on prefix
        if client_name.startswith("jetson"):
            return f"192.168.100.{10 + num}"
        elif client_name.startswith("guest"):
            return f"192.168.100.{200 + num}"
        elif client_name.startswith("server"):
            return f"192.168.100.{num}"
        else:
            # Generic pattern for other prefixes
            return f"192.168.100.{100 + num}"
            
    except (ValueError, AttributeError):
        return None

@app.get("/nebula/peers")
def get_nebula_peers() -> JSONResponse:
    """
    Get information about active Nebula VPN peers.
    
    Executes nebula command to retrieve peer information including
    VPN IPs, public endpoints, and latency measurements.
    
    Returns:
        JSONResponse: Dictionary containing:
            - online_peers: List of peer information
            - peer_count: Total number of active peers
            - error: Error message if command fails
    """
    try:
        # Execute nebula command with timeout to prevent hanging
        result = subprocess.check_output(
            ["nebula", "-config", CONFIG_FILE_PATH, "-test"],
            stderr=subprocess.PIPE,
            timeout=10  # 10 second timeout
        ).decode()

        peers = _parse_peer_output(result)
        
        # Limit number of peers returned for performance
        if len(peers) > MAX_PEERS_DISPLAY:
            peers = peers[:MAX_PEERS_DISPLAY]
            
        return JSONResponse({
            "online_peers": peers,
            "peer_count": len(peers),
            "truncated": len(peers) == MAX_PEERS_DISPLAY,
            "timestamp": subprocess.check_output(["date", "+%Y-%m-%d %H:%M:%S"]).decode().strip()
        })
        
    except subprocess.TimeoutExpired:
        return JSONResponse({
            "error": "Nebula command timed out",
            "online_peers": [],
            "peer_count": 0
        })
    except subprocess.CalledProcessError as e:
        return JSONResponse({
            "error": f"Nebula command failed: {e.stderr.decode() if e.stderr else str(e)}",
            "online_peers": [],
            "peer_count": 0
        })
    except FileNotFoundError:
        return JSONResponse({
            "error": "Nebula binary not found in PATH",
            "online_peers": [],
            "peer_count": 0
        })
    except Exception as e:
        return JSONResponse({
            "error": f"Unexpected error: {str(e)}",
            "online_peers": [],
            "peer_count": 0
        })


def _parse_peer_output(output: str) -> List[Dict[str, str]]:
    """
    Parse nebula peer command output into structured data.
    
    Args:
        output: Raw output from nebula -test command
        
    Returns:
        List of peer dictionaries with parsed information
    """
    peers = []
    peer_blocks = output.strip().split("\n\n")
    
    for block in peer_blocks:
        if not block.strip():
            continue
            
        peer = {}
        for line in block.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
                
            # Parse different types of peer information
            if "Remote VPN IP" in line:
                peer["vpn_ip"] = line.split(":", 1)[1].strip()
            elif "Remote" in line and "VPN" not in line:
                peer["public_endpoint"] = line.split(":", 1)[1].strip()
            elif "Latency" in line:
                peer["latency"] = line.split(":", 1)[1].strip()
            elif "Relay" in line:
                peer["relay"] = line.split(":", 1)[1].strip()
            elif "Lighthouse" in line:
                peer["is_lighthouse"] = "true" in line.lower()
                
        # Only add peers with at least VPN IP information
        if peer.get("vpn_ip"):
            peers.append(peer)
            
    return peers

@app.get("/download/{client_id}")
def download_nebula_bundle(
    client_id: str,
    token: str = Query(..., description="Authentication token for client access"),
    platform: str = Query(..., description="Target platform (macos-arm64, linux-amd64, linux-arm64)"),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> FileResponse:
    """
    Download a complete Nebula VPN client bundle.
    
    Creates a zip file containing all necessary files for a client to connect:
    - Client certificate and private key
    - Configuration file
    - Platform-specific Nebula binary
    - Startup script (run.sh)
    
    Usage example:
        curl -O -G \
          -d "token=mysecrettoken" \
          -d "platform=macos-arm64" \
          http://your-server:8000/download/guest01
        
        unzip guest01.zip && ./run.sh
    
    Args:
        client_id: Unique identifier for the client (e.g., 'jetson01', 'guest01')
        token: Authentication token (must be in client's tokens.json)
        platform: Target platform identifier
        
    Returns:
        FileResponse: Zip file containing complete client bundle
        
    Raises:
        HTTPException: Various error codes for different failure scenarios
    """
    print(f"\n\n=== Starting download request for client: {client_id}, platform: {platform} ===\n")
    
    # Use a persistent temporary directory to avoid cleanup issues
    persistent_tmp_dir = None
    
    try:
        # Input validation and security checks
        _validate_client_id(client_id)
        
        client_path = CLIENTS_DIR / client_id
        print(f"Client path: {client_path}, exists: {client_path.exists()}")
        
        if not client_path.exists():
            raise HTTPException(status_code=404, detail=f"Client directory not found: {client_path}")

        # Verify client has valid token using unified verification method
        print(f"Verifying token for client: {client_id}")
        _verify_client_specific_token(client_path, token)
        
        # Validate platform and check binary availability
        print(f"Validating platform: {platform}")
        _validate_platform(platform)
        
        # Verify all required files exist before creating bundle
        print(f"Checking required files for client: {client_id}")
        required_files = _check_required_files(client_path, client_id)
        
        # Create a persistent temporary directory that won't be automatically cleaned up
        # This helps prevent the file from being deleted before it can be served
        persistent_tmp_dir = tempfile.mkdtemp(prefix=f"nebula-{client_id}-")
        print(f"Created persistent temporary directory: {persistent_tmp_dir}")
        
        # Create the bundle
        bundle_path = _create_client_bundle(persistent_tmp_dir, client_path, client_id, platform, required_files)
        
        # Verify the bundle exists and is readable before attempting to serve it
        if not bundle_path.exists():
            raise Exception(f"Bundle file was not created at expected path: {bundle_path}")
            
        if not os.access(bundle_path, os.R_OK):
            raise Exception(f"Bundle file is not readable: {bundle_path}")
            
        print(f"Bundle created successfully at: {bundle_path}, size: {bundle_path.stat().st_size} bytes")
        
        # Create the response
        response = FileResponse(
            path=bundle_path, 
            filename=f"{client_id}.zip", 
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={client_id}.zip",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
        
        # Register a background task to clean up the temporary directory after the response is sent
        def cleanup_temp_dir():
            try:
                if persistent_tmp_dir and os.path.exists(persistent_tmp_dir):
                    print(f"Cleaning up temporary directory: {persistent_tmp_dir}")
                    shutil.rmtree(persistent_tmp_dir, ignore_errors=True)
            except Exception as e:
                print(f"Error cleaning up temporary directory: {str(e)}")
        
        # Add the cleanup task to background tasks
        background_tasks.add_task(cleanup_temp_dir)
        
        print(f"Returning file response for client: {client_id}")
        return response
            
    except HTTPException as he:
        # Re-raise HTTP exceptions as-is
        print(f"HTTP Exception: {he.status_code} - {he.detail}")
        raise
    except Exception as e:
        # Log the full error for debugging
        import traceback
        error_details = f"Unexpected error in download_nebula_bundle: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(error_details)  # This will appear in server logs
        
        # Clean up the temporary directory if it exists
        if persistent_tmp_dir and os.path.exists(persistent_tmp_dir):
            try:
                shutil.rmtree(persistent_tmp_dir, ignore_errors=True)
            except Exception as cleanup_error:
                print(f"Error cleaning up temporary directory: {str(cleanup_error)}")
        
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )


def _validate_client_id(client_id: str) -> None:
    """
    Validate client ID for security and format compliance.
    
    Args:
        client_id: Client identifier to validate
        
    Raises:
        HTTPException: If client ID is invalid
    """
    # Security: Prevent path traversal and injection attacks
    if not re.match(r'^[a-zA-Z0-9_-]+$', client_id):
        raise HTTPException(
            status_code=400, 
            detail="Client ID must contain only alphanumeric characters, hyphens, and underscores"
        )
    
    if len(client_id) > 50:
        raise HTTPException(status_code=400, detail="Client ID too long")
    
    if client_id.startswith('.') or client_id.startswith('-'):
        raise HTTPException(status_code=400, detail="Invalid client ID format")





def _validate_platform(platform: str) -> None:
    """
    Validate platform identifier and check binary availability.
    
    Args:
        platform: Platform identifier to validate
        
    Raises:
        HTTPException: If platform is unsupported or binary missing
    """
    if platform not in BINARIES:
        available_platforms = list(BINARIES.keys())
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported platform: {platform}. Available: {available_platforms}"
        )
    
    binary_path = BINARIES[platform]
    if not binary_path.exists():
        raise HTTPException(
            status_code=500, 
            detail=f"Binary not found for platform {platform}: {binary_path}"
        )
    
    if not os.access(binary_path, os.R_OK):
        raise HTTPException(
            status_code=500, 
            detail=f"Binary not readable for platform {platform}"
        )


def _check_required_files(client_path: Path, client_id: str) -> Dict[str, Path]:
    """
    Check that all required client files exist and are readable.
    
    Args:
        client_path: Path to client directory
        client_id: Client identifier
        
    Returns:
        Dictionary mapping file types to their paths
        
    Raises:
        HTTPException: If required files are missing or unreadable
    """
    required_files = {
        "certificate": client_path / f"{client_id}.crt",
        "private_key": client_path / f"{client_id}.key",
        "config": client_path / "config.yml"
    }
    
    missing_files = []
    for file_type, file_path in required_files.items():
        if not file_path.exists():
            missing_files.append(f"{file_type} ({file_path.name})")
        elif not os.access(file_path, os.R_OK):
            raise HTTPException(
                status_code=500, 
                detail=f"Cannot read {file_type} file: {file_path.name}"
            )
    
    if missing_files:
        raise HTTPException(
            status_code=404, 
            detail=f"Missing required files: {', '.join(missing_files)}"
        )
    
    return required_files


def _create_client_bundle(tmpdir: str, client_path: Path, client_id: str, platform: str, required_files: Dict[str, Path]) -> Path:
    """
    Create the client bundle zip file.
    
    Args:
        tmpdir: Temporary directory path
        client_path: Path to client directory
        client_id: Client identifier
        platform: Target platform
        required_files: Dictionary of required file paths
        
    Returns:
        Path to created zip file
        
    Raises:
        Exception: If any step in the bundle creation process fails
    """
    tmp = Path(tmpdir)
    zip_path = tmp / f"{client_id}.zip"
    
    print(f"Creating bundle in temporary directory: {tmp}")
    print(f"Temporary directory exists: {tmp.exists()}, is writable: {os.access(tmp, os.W_OK)}")
    
    # Copy required files with error handling
    try:
        # Copy client files
        for file_type, src_path in required_files.items():
            dst_path = tmp / src_path.name
            print(f"Copying {file_type} from {src_path} to {dst_path}")
            if not src_path.exists():
                raise Exception(f"Source file does not exist: {src_path}")
            if not os.access(src_path, os.R_OK):
                raise Exception(f"Source file is not readable: {src_path}")
            
            shutil.copy2(src_path, dst_path)  # copy2 preserves metadata
            print(f"Successfully copied {file_type} file")
            
        # Copy and prepare Nebula binary
        binary_src = BINARIES[platform]
        binary_dst = tmp / "nebula"
        print(f"Copying binary from {binary_src} to {binary_dst}")
        if not binary_src.exists():
            raise Exception(f"Binary does not exist: {binary_src}")
        if not os.access(binary_src, os.R_OK):
            raise Exception(f"Binary is not readable: {binary_src}")
            
        shutil.copy2(binary_src, binary_dst)
        binary_dst.chmod(0o755)  # Ensure executable permissions
        print(f"Successfully copied binary with executable permissions")
        
    except (OSError, PermissionError) as e:
        raise Exception(f"Failed to copy files: {str(e)}")
    
    try:
        # Create enhanced startup script
        print("Creating startup script")
        startup_script = _generate_startup_script(client_id)
        script_path = tmp / "run.sh"
        script_path.write_text(startup_script, encoding='utf-8')
        script_path.chmod(0o755)
        
        # Create README with usage instructions
        print("Creating README file")
        readme_content = _generate_readme(client_id, platform)
        (tmp / "README.txt").write_text(readme_content, encoding='utf-8')
    except Exception as e:
        raise Exception(f"Failed to create script or README files: {str(e)}")
    
    try:
        # Create zip bundle with optimal compression
        print(f"Creating zip file at {zip_path}")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
            for file_path in tmp.iterdir():
                if file_path != zip_path:  # Don't include the zip file itself
                    print(f"Adding {file_path.name} to zip")
                    zipf.write(file_path, arcname=file_path.name)
        
        # Verify the zip file was created successfully
        if not zip_path.exists():
            raise Exception(f"Zip file was not created: {zip_path}")
        if not os.access(zip_path, os.R_OK):
            raise Exception(f"Zip file is not readable: {zip_path}")
            
        print(f"Successfully created zip file: {zip_path}, size: {zip_path.stat().st_size} bytes")
        return zip_path
        
    except Exception as e:
        raise Exception(f"Failed to create zip file: {str(e)}")


def _generate_startup_script(client_id: str) -> str:
    """
    Generate an enhanced startup script for the client.
    
    Args:
        client_id: Client identifier
        
    Returns:
        Startup script content
    """
    return f'''#!/bin/bash
# Nebula VPN Client Startup Script
# Client: {client_id}
# Generated: $(date)

set -e  # Exit on any error

echo "Starting Nebula VPN client: {client_id}"
echo "Timestamp: $(date)"

# Make binary executable
chmod +x nebula

# Check if config file exists
if [ ! -f "config.yml" ]; then
    echo "Error: config.yml not found"
    exit 1
fi

# Check if certificate files exist
if [ ! -f "{client_id}.crt" ] || [ ! -f "{client_id}.key" ]; then
    echo "Error: Certificate files not found"
    exit 1
fi

echo "Starting Nebula with config: config.yml"
echo "Press Ctrl+C to stop"

# Start Nebula VPN
./nebula -config config.yml
'''


def _generate_readme(client_id: str, platform: str) -> str:
    """
    Generate README file with usage instructions.
    
    Args:
        client_id: Client identifier
        platform: Target platform
        
    Returns:
        README content
    """
    return f'''Nebula VPN Client Bundle
========================

Client ID: {client_id}
Platform: {platform}
Generated: $(date)

Contents:
---------
- nebula: VPN client binary
- config.yml: Network configuration
- {client_id}.crt: Client certificate
- {client_id}.key: Client private key
- run.sh: Startup script
- README.txt: This file

Quick Start:
-----------
1. Extract this zip file
2. Open terminal in the extracted directory
3. Run: ./run.sh
4. Press Ctrl+C to stop

Manual Usage:
------------
./nebula -config config.yml

Troubleshooting:
---------------
- Ensure all files are present
- Check that nebula binary has execute permissions
- Verify network connectivity
- Check firewall settings for UDP traffic

For support, contact your network administrator.
'''
# ============================================================================
# PUBLIC FILE DOWNLOAD ENDPOINT
# ============================================================================

@app.get("/public/{folder}/{filename}")
def download_public_file(
    folder: str,
    filename: str,
    token: str = Query(..., description="Authentication token for access"),
) -> FileResponse:
    """
    Download a file from the public directory.
    
    Args:
        folder: The folder within the public directory
        filename: Name of the file to download
        token: Authentication token (must be in VALID_TOKENS)
        
    Returns:
        FileResponse: The requested file as a download
        
    Raises:
        HTTPException: 401 if token invalid, 404 if file not found, 400 if path traversal detected
        
    Usage example:
        curl -O -G \
          -d "token=jetsonsupertoken" \
          http://your-server:8000/public/downloads/example.zip
    """
    # Verify token against global tokens
    if token not in VALID_TOKENS:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Security: Prevent path traversal attacks
    if ".." in folder or ".." in filename or folder.startswith("/") or filename.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid path")
    
    # Construct the file path
    file_path = PUBLIC_DIR / folder / filename
    
    # Ensure the resolved path is still within PUBLIC_DIR
    try:
        resolved_path = file_path.resolve()
        if not str(resolved_path).startswith(str(PUBLIC_DIR.resolve())):
            raise ValueError("Path traversal detected")
    except (ValueError, RuntimeError):
        raise HTTPException(status_code=400, detail="Invalid file path")
    
    # Check if file exists
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Check if file is readable
    if not os.access(file_path, os.R_OK):
        raise HTTPException(status_code=403, detail="Permission denied")
    
    # Return the file as a download
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )

# ============================================================================
# SERVER STARTUP AND USAGE INSTRUCTIONS
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🚀 Nebula VPN Management API Server")
    print("="*60)
    print(f"📁 Base Directory: {BASE_DIR}")
    print(f"👥 Clients Directory: {CLIENTS_DIR}")
    print(f"🔧 Available Platforms: {list(BINARIES.keys())}")
    print(f"📚 API Documentation: http://localhost:8000/docs")
    print(f"🔍 Alternative Docs: http://localhost:8000/redoc")
    print("="*60)
    print("\n🔗 Available Endpoints:")
    print("  • GET /nebula/status - VPN status information")
    print("  • GET /nebula/members - List all clients")
    print("  • GET /nebula/peers - Active peer information")
    print("  • GET /nebula/{client}/{filename} - Serve client files")
    print("  • GET /download/{client_id} - Download client bundle")
    print("\n💡 Usage Examples:")
    print("  # Check VPN status (requires auth):")
    print("  curl -H 'Authorization: Bearer jetsonsupertoken' http://localhost:8000/nebula/status")
    print("\n  # List all clients:")
    print("  curl http://localhost:8000/nebula/members")
    print("\n  # Download client bundle:")
    print("  curl -O -G -d 'token=mysecrettoken' -d 'platform=linux-amd64' \\")
    print("       http://localhost:8000/download/guest01")
    print("\n  # Extract and run:")
    print("  unzip guest01.zip && ./run.sh")
    print("\n" + "="*60)
    
    # Start the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        access_log=True,
        log_level="info"
    )

# ============================================================================
# PRODUCTION DEPLOYMENT COMMANDS
# ============================================================================

# Development server:
#pip3 install fastapi uvicorn
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Production server (background):
# nohup uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4 > api.log 2>&1 &

# With Gunicorn (recommended for production):
# pip install gunicorn
# gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Docker deployment:
# docker run -d -p 8000:8000 -v /path/to/serverdata:/home/cmpe/Documents/serverdata your-image

# ============================================================================
# SECURITY CONSIDERATIONS
# ============================================================================

# TODO: Production security improvements:
# 1. Move VALID_TOKENS to environment variables or secure key management
# 2. Implement rate limiting (e.g., slowapi)
# 3. Add request logging and monitoring
# 4. Use HTTPS in production with proper certificates
# 5. Implement token expiration and rotation
# 6. Add input sanitization and validation middleware
# 7. Configure CORS properly for production origins
# 8. Add health check endpoints for load balancers
# 9. Implement proper error handling and logging
# 10. Add metrics and monitoring (Prometheus, etc.)
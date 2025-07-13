#!/usr/bin/env python3
"""
Simple MCP Server for Ollama Gradio Client
Optimized for edge devices (Jetson Orin Nano, Apple Silicon)

This server provides basic tools that can be used with the Ollama MCP client.
Includes image generation, file operations, and system information tools.
"""

import json
import sys
import io
import os
import time
import platform
import subprocess
from typing import Dict, Any, Optional

# Ensure proper encoding for stdio
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    print("Error: MCP not available. Install with: pip install mcp", file=sys.stderr)
    sys.exit(1)

# Try to import optional dependencies
try:
    from gradio_client import Client
    GRADIO_CLIENT_AVAILABLE = True
except ImportError:
    GRADIO_CLIENT_AVAILABLE = False
    print("Warning: gradio_client not available. Image generation will be limited.", file=sys.stderr)

# Try to import networking and security libraries
try:
    import socket
    import urllib.request
    import urllib.parse
    import urllib.error
    import requests
    NETWORKING_AVAILABLE = True
except ImportError:
    NETWORKING_AVAILABLE = False
    print("Warning: Some networking libraries not available.", file=sys.stderr)

try:
    import scapy.all as scapy
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("Warning: Scapy not available. Install with: pip install scapy", file=sys.stderr)

# Initialize MCP server
mcp = FastMCP("edge_ai_tools")

@mcp.tool()
async def generate_simple_image(prompt: str, style: str = "realistic") -> str:
    """
    Generate a simple image description or ASCII art.
    
    Args:
        prompt: Text prompt describing the image
        style: Style of the image (realistic, cartoon, ascii)
    """
    try:
        # For edge devices, we'll create a simple text-based image description
        # instead of actual image generation to save resources
        
        if style.lower() == "ascii":
            # Simple ASCII art generation
            ascii_art = generate_ascii_art(prompt)
            return json.dumps({
                "type": "ascii_art",
                "content": ascii_art,
                "prompt": prompt,
                "message": f"Generated ASCII art for: {prompt}"
            })
        
        # If gradio_client is available, try to use a lightweight image generation service
        if GRADIO_CLIENT_AVAILABLE:
            try:
                # Use a lightweight image generation space
                client = Client("https://hf.co/spaces/stabilityai/stable-diffusion")
                result = client.predict(
                    prompt,
                    "Euler a",  # sampler
                    25,  # steps (reduced for edge devices)
                    7.5,  # guidance scale
                    512,  # width (reduced for edge devices)
                    512,  # height (reduced for edge devices)
                    api_name="/txt2img"
                )
                
                if result and len(result) > 0:
                    return json.dumps({
                        "type": "image_url",
                        "url": result[0] if isinstance(result, list) else str(result),
                        "prompt": prompt,
                        "message": f"Generated image for: {prompt}"
                    })
            except Exception as e:
                print(f"Image generation failed: {e}", file=sys.stderr)
        
        # Fallback: return a detailed description
        description = f"Image Description for '{prompt}':\n\n"
        description += f"Style: {style}\n"
        description += f"A {style} image showing {prompt}. "
        description += "This would be a detailed visual representation "
        description += "optimized for the requested style and content."
        
        return json.dumps({
            "type": "description",
            "content": description,
            "prompt": prompt,
            "message": f"Generated description for: {prompt} (actual image generation not available)"
        })
        
    except Exception as e:
        return json.dumps({
            "type": "error",
            "message": f"Error generating image: {str(e)}"
        })

def generate_ascii_art(prompt: str) -> str:
    """Generate simple ASCII art based on prompt"""
    prompt_lower = prompt.lower()
    
    if "cat" in prompt_lower:
        return """
    /\_/\  
   ( o.o ) 
    > ^ <
        """
    elif "dog" in prompt_lower:
        return """
    / \   / \
   (   @ @   )
    \   o   /
     \\___//
        """
    elif "house" in prompt_lower:
        return """
      /\
     /  \
    /____\
    |    |
    | [] |
    |____|    
        """
    elif "tree" in prompt_lower:
        return """
       🌲
      🌲🌲🌲
     🌲🌲🌲🌲🌲
        |||
        |||
        """
    else:
        return f"""
    ┌─────────────────┐
    │  {prompt[:15]:<15} │
    │                 │
    │   ASCII ART     │
    │  PLACEHOLDER    │
    └─────────────────┘
        """

@mcp.tool()
async def get_system_info() -> str:
    """
    Get detailed system information for the edge device.
    """
    try:
        info = {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Try to get GPU information
        try:
            # Check for NVIDIA GPU
            nvidia_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if nvidia_result.returncode == 0:
                info["nvidia_gpu"] = nvidia_result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Check for Jetson-specific info
        try:
            jetson_result = subprocess.run(
                ["jetson_release", "-v"],
                capture_output=True, text=True, timeout=5
            )
            if jetson_result.returncode == 0:
                info["jetson_info"] = jetson_result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Check memory usage
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = f.read()
                for line in meminfo.split("\n"):
                    if "MemTotal" in line:
                        info["total_memory"] = line.split()[1] + " kB"
                    elif "MemAvailable" in line:
                        info["available_memory"] = line.split()[1] + " kB"
        except FileNotFoundError:
            pass
        
        return json.dumps({
            "type": "system_info",
            "data": info,
            "message": "System information retrieved successfully"
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "type": "error",
            "message": f"Error getting system info: {str(e)}"
        })

@mcp.tool()
async def create_file(filename: str, content: str, directory: str = "./") -> str:
    """
    Create a text file with specified content.
    
    Args:
        filename: Name of the file to create
        content: Content to write to the file
        directory: Directory to create the file in (default: current directory)
    """
    try:
        # Security check: prevent directory traversal
        if ".." in filename or ".." in directory:
            return json.dumps({
                "type": "error",
                "message": "Invalid path: directory traversal not allowed"
            })
        
        # Ensure directory exists
        os.makedirs(directory, exist_ok=True)
        
        # Create full path
        filepath = os.path.join(directory, filename)
        
        # Write file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        
        return json.dumps({
            "type": "file_created",
            "filepath": filepath,
            "size": len(content),
            "message": f"File '{filename}' created successfully in '{directory}'"
        })
        
    except Exception as e:
        return json.dumps({
            "type": "error",
            "message": f"Error creating file: {str(e)}"
        })

@mcp.tool()
async def read_file(filepath: str) -> str:
    """
    Read content from a text file.
    
    Args:
        filepath: Path to the file to read
    """
    try:
        # Security check: prevent directory traversal
        if ".." in filepath:
            return json.dumps({
                "type": "error",
                "message": "Invalid path: directory traversal not allowed"
            })
        
        # Check if file exists
        if not os.path.exists(filepath):
            return json.dumps({
                "type": "error",
                "message": f"File not found: {filepath}"
            })
        
        # Read file
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        return json.dumps({
            "type": "file_content",
            "filepath": filepath,
            "content": content,
            "size": len(content),
            "message": f"File '{filepath}' read successfully"
        })
        
    except Exception as e:
        return json.dumps({
            "type": "error",
            "message": f"Error reading file: {str(e)}"
        })

@mcp.tool()
async def list_directory(directory: str = "./") -> str:
    """
    List contents of a directory.
    
    Args:
        directory: Directory to list (default: current directory)
    """
    try:
        # Security check: prevent directory traversal
        if ".." in directory:
            return json.dumps({
                "type": "error",
                "message": "Invalid path: directory traversal not allowed"
            })
        
        # Check if directory exists
        if not os.path.exists(directory):
            return json.dumps({
                "type": "error",
                "message": f"Directory not found: {directory}"
            })
        
        # List directory contents
        items = []
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            item_info = {
                "name": item,
                "type": "directory" if os.path.isdir(item_path) else "file",
                "size": os.path.getsize(item_path) if os.path.isfile(item_path) else None
            }
            items.append(item_info)
        
        return json.dumps({
            "type": "directory_listing",
            "directory": directory,
            "items": items,
            "count": len(items),
            "message": f"Listed {len(items)} items in '{directory}'"
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "type": "error",
            "message": f"Error listing directory: {str(e)}"
        })

@mcp.tool()
async def network_ping(host: str, count: int = 4, timeout: int = 10) -> str:
    """
    Ping a host to test network connectivity.
    
    Args:
        host: Hostname or IP address to ping
        count: Number of ping packets to send (default: 4)
        timeout: Timeout in seconds (default: 10)
    """
    try:
        # Validate input
        if not host or len(host) > 253:
            return json.dumps({
                "type": "error",
                "message": "Invalid hostname or IP address"
            })
        
        # Run ping command
        result = subprocess.run(
            ["ping", "-c", str(count), host],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        return json.dumps({
            "type": "ping_result",
            "host": host,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0,
            "message": f"Ping to {host} {'successful' if result.returncode == 0 else 'failed'}"
        })
        
    except subprocess.TimeoutExpired:
        return json.dumps({
            "type": "error",
            "message": f"Ping to {host} timed out after {timeout} seconds"
        })
    except Exception as e:
        return json.dumps({
            "type": "error",
            "message": f"Error pinging {host}: {str(e)}"
        })

@mcp.tool()
async def network_traceroute(host: str, max_hops: int = 30, timeout: int = 30) -> str:
    """
    Trace the network route to a host.
    
    Args:
        host: Hostname or IP address to trace
        max_hops: Maximum number of hops (default: 30)
        timeout: Timeout in seconds (default: 30)
    """
    try:
        if not host or len(host) > 253:
            return json.dumps({
                "type": "error",
                "message": "Invalid hostname or IP address"
            })
        
        # Try traceroute first, then tracepath as fallback
        for cmd in ["traceroute", "tracepath"]:
            try:
                result = subprocess.run(
                    [cmd, "-m", str(max_hops), host],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                return json.dumps({
                    "type": "traceroute_result",
                    "host": host,
                    "command": cmd,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "message": f"Traceroute to {host} completed"
                })
            except FileNotFoundError:
                continue
        
        return json.dumps({
            "type": "error",
            "message": "Neither traceroute nor tracepath command available"
        })
        
    except subprocess.TimeoutExpired:
        return json.dumps({
            "type": "error",
            "message": f"Traceroute to {host} timed out after {timeout} seconds"
        })
    except Exception as e:
        return json.dumps({
            "type": "error",
            "message": f"Error tracing route to {host}: {str(e)}"
        })

@mcp.tool()
async def network_scan_ports(host: str, ports: str = "22,80,443,8080", timeout: int = 30) -> str:
    """
    Scan specific ports on a host using nmap or basic socket connection.
    
    Args:
        host: Hostname or IP address to scan
        ports: Comma-separated list of ports (default: "22,80,443,8080")
        timeout: Timeout in seconds (default: 30)
    """
    try:
        if not host or len(host) > 253:
            return json.dumps({
                "type": "error",
                "message": "Invalid hostname or IP address"
            })
        
        # Validate and parse ports
        try:
            port_list = [int(p.strip()) for p in ports.split(",") if p.strip()]
            if not port_list or any(p < 1 or p > 65535 for p in port_list):
                raise ValueError("Invalid port numbers")
        except ValueError:
            return json.dumps({
                "type": "error",
                "message": "Invalid port specification. Use comma-separated numbers (1-65535)"
            })
        
        # Try nmap first
        try:
            result = subprocess.run(
                ["nmap", "-p", ports, host],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return json.dumps({
                "type": "port_scan_result",
                "host": host,
                "ports": ports,
                "method": "nmap",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "message": f"Port scan of {host} completed using nmap"
            })
        except FileNotFoundError:
            # Fallback to basic socket scanning
            if not NETWORKING_AVAILABLE:
                return json.dumps({
                    "type": "error",
                    "message": "Neither nmap nor socket library available for port scanning"
                })
            
            scan_results = []
            for port in port_list:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(2)
                    result = sock.connect_ex((host, port))
                    sock.close()
                    scan_results.append({
                        "port": port,
                        "status": "open" if result == 0 else "closed"
                    })
                except Exception as e:
                    scan_results.append({
                        "port": port,
                        "status": "error",
                        "error": str(e)
                    })
            
            return json.dumps({
                "type": "port_scan_result",
                "host": host,
                "method": "socket",
                "results": scan_results,
                "message": f"Port scan of {host} completed using socket connections"
            })
        
    except subprocess.TimeoutExpired:
        return json.dumps({
            "type": "error",
            "message": f"Port scan of {host} timed out after {timeout} seconds"
        })
    except Exception as e:
        return json.dumps({
            "type": "error",
            "message": f"Error scanning ports on {host}: {str(e)}"
        })

@mcp.tool()
async def http_request(url: str, method: str = "GET", headers: str = "", data: str = "", timeout: int = 30) -> str:
    """
    Send HTTP request to a URL.
    
    Args:
        url: URL to send request to
        method: HTTP method (GET, POST, PUT, DELETE, etc.)
        headers: JSON string of headers (optional)
        data: Request body data (optional)
        timeout: Timeout in seconds (default: 30)
    """
    try:
        if not NETWORKING_AVAILABLE:
            return json.dumps({
                "type": "error",
                "message": "Networking libraries not available"
            })
        
        # Validate URL
        if not url.startswith(('http://', 'https://')):
            return json.dumps({
                "type": "error",
                "message": "URL must start with http:// or https://"
            })
        
        # Parse headers
        request_headers = {}
        if headers:
            try:
                request_headers = json.loads(headers)
            except json.JSONDecodeError:
                return json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format for headers"
                })
        
        # Make request
        try:
            import requests
            response = requests.request(
                method=method.upper(),
                url=url,
                headers=request_headers,
                data=data if data else None,
                timeout=timeout
            )
            
            return json.dumps({
                "type": "http_response",
                "url": url,
                "method": method.upper(),
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": response.text[:1000],  # Limit content size
                "content_length": len(response.text),
                "message": f"HTTP {method.upper()} request to {url} completed"
            })
        except requests.RequestException as e:
            return json.dumps({
                "type": "error",
                "message": f"HTTP request failed: {str(e)}"
            })
        
    except Exception as e:
        return json.dumps({
            "type": "error",
            "message": f"Error making HTTP request: {str(e)}"
        })

@mcp.tool()
async def send_udp_packet(host: str, port: int, message: str, timeout: int = 10) -> str:
    """
    Send a UDP packet to a host.
    
    Args:
        host: Hostname or IP address
        port: Port number (1-65535)
        message: Message to send
        timeout: Timeout in seconds (default: 10)
    """
    try:
        if not NETWORKING_AVAILABLE:
            return json.dumps({
                "type": "error",
                "message": "Socket library not available"
            })
        
        # Validate inputs
        if not host or port < 1 or port > 65535:
            return json.dumps({
                "type": "error",
                "message": "Invalid host or port number"
            })
        
        # Create UDP socket and send packet
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)
        
        try:
            # Send packet
            bytes_sent = sock.sendto(message.encode('utf-8'), (host, port))
            
            # Try to receive response (optional)
            try:
                sock.settimeout(2)  # Short timeout for response
                response, addr = sock.recvfrom(1024)
                response_data = response.decode('utf-8', errors='ignore')
            except socket.timeout:
                response_data = None
                addr = None
            
            sock.close()
            
            return json.dumps({
                "type": "udp_send_result",
                "host": host,
                "port": port,
                "bytes_sent": bytes_sent,
                "response": response_data,
                "response_from": str(addr) if addr else None,
                "message": f"UDP packet sent to {host}:{port}"
            })
            
        except socket.error as e:
            sock.close()
            return json.dumps({
                "type": "error",
                "message": f"UDP send failed: {str(e)}"
            })
        
    except Exception as e:
        return json.dumps({
            "type": "error",
            "message": f"Error sending UDP packet: {str(e)}"
        })

@mcp.tool()
async def tcp_connect(host: str, port: int, message: str = "", timeout: int = 10) -> str:
    """
    Establish TCP connection and optionally send data.
    
    Args:
        host: Hostname or IP address
        port: Port number (1-65535)
        message: Optional message to send
        timeout: Timeout in seconds (default: 10)
    """
    try:
        if not NETWORKING_AVAILABLE:
            return json.dumps({
                "type": "error",
                "message": "Socket library not available"
            })
        
        # Validate inputs
        if not host or port < 1 or port > 65535:
            return json.dumps({
                "type": "error",
                "message": "Invalid host or port number"
            })
        
        # Create TCP socket and connect
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        
        try:
            # Connect
            start_time = time.time()
            sock.connect((host, port))
            connect_time = time.time() - start_time
            
            response_data = None
            if message:
                # Send message
                sock.send(message.encode('utf-8'))
                
                # Try to receive response
                try:
                    sock.settimeout(2)  # Short timeout for response
                    response = sock.recv(1024)
                    response_data = response.decode('utf-8', errors='ignore')
                except socket.timeout:
                    response_data = "No response received"
            
            sock.close()
            
            return json.dumps({
                "type": "tcp_connect_result",
                "host": host,
                "port": port,
                "connect_time": round(connect_time * 1000, 2),  # ms
                "message_sent": bool(message),
                "response": response_data,
                "message": f"TCP connection to {host}:{port} successful"
            })
            
        except socket.error as e:
            sock.close()
            return json.dumps({
                "type": "error",
                "message": f"TCP connection failed: {str(e)}"
            })
        
    except Exception as e:
        return json.dumps({
            "type": "error",
            "message": f"Error establishing TCP connection: {str(e)}"
        })

@mcp.tool()
async def network_interface_info() -> str:
    """
    Get network interface information.
    """
    try:
        # Try multiple commands to get interface info
        commands = [
            ["ip", "addr", "show"],
            ["ifconfig"],
            ["ip", "link", "show"]
        ]
        
        for cmd in commands:
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    return json.dumps({
                        "type": "network_interface_info",
                        "command": " ".join(cmd),
                        "output": result.stdout,
                        "message": "Network interface information retrieved"
                    })
            except FileNotFoundError:
                continue
        
        return json.dumps({
            "type": "error",
            "message": "No suitable network interface command available"
        })
        
    except Exception as e:
        return json.dumps({
            "type": "error",
            "message": f"Error getting network interface info: {str(e)}"
        })

@mcp.tool()
async def dns_lookup(hostname: str, record_type: str = "A", timeout: int = 10) -> str:
    """
    Perform DNS lookup for a hostname.
    
    Args:
        hostname: Hostname to lookup
        record_type: DNS record type (A, AAAA, MX, NS, TXT, etc.)
        timeout: Timeout in seconds (default: 10)
    """
    try:
        if not hostname:
            return json.dumps({
                "type": "error",
                "message": "Hostname cannot be empty"
            })
        
        # Try dig first, then nslookup
        for cmd, args in [("dig", ["+short", "-t", record_type, hostname]), 
                         ("nslookup", ["-type=" + record_type, hostname])]:
            try:
                result = subprocess.run(
                    [cmd] + args,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                return json.dumps({
                    "type": "dns_lookup_result",
                    "hostname": hostname,
                    "record_type": record_type,
                    "command": cmd,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "message": f"DNS lookup for {hostname} ({record_type}) completed"
                })
            except FileNotFoundError:
                continue
        
        return json.dumps({
            "type": "error",
            "message": "Neither dig nor nslookup command available"
        })
        
    except subprocess.TimeoutExpired:
        return json.dumps({
            "type": "error",
            "message": f"DNS lookup for {hostname} timed out"
        })
    except Exception as e:
        return json.dumps({
            "type": "error",
            "message": f"Error performing DNS lookup: {str(e)}"
        })

@mcp.tool()
async def packet_capture(interface: str = "any", count: int = 10, filter_expr: str = "", timeout: int = 30) -> str:
    """
    Capture network packets using tcpdump or tshark.
    
    Args:
        interface: Network interface to capture on (default: "any")
        count: Number of packets to capture (default: 10)
        filter_expr: BPF filter expression (optional)
        timeout: Timeout in seconds (default: 30)
    """
    try:
        # Validate inputs
        if count < 1 or count > 100:
            return json.dumps({
                "type": "error",
                "message": "Packet count must be between 1 and 100"
            })
        
        # Try tcpdump first, then tshark
        commands = [
            ["tcpdump", "-i", interface, "-c", str(count), "-n"],
            ["tshark", "-i", interface, "-c", str(count), "-n"]
        ]
        
        # Add filter if provided
        if filter_expr:
            for cmd in commands:
                cmd.append(filter_expr)
        
        for cmd in commands:
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                return json.dumps({
                    "type": "packet_capture_result",
                    "interface": interface,
                    "count": count,
                    "filter": filter_expr,
                    "command": cmd[0],
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "message": f"Captured {count} packets on {interface} using {cmd[0]}"
                })
            except FileNotFoundError:
                continue
        
        return json.dumps({
            "type": "error",
            "message": "Neither tcpdump nor tshark available for packet capture"
        })
        
    except subprocess.TimeoutExpired:
        return json.dumps({
            "type": "error",
            "message": f"Packet capture timed out after {timeout} seconds"
        })
    except Exception as e:
        return json.dumps({
            "type": "error",
            "message": f"Error capturing packets: {str(e)}"
        })

@mcp.tool()
async def security_scan_vulnerabilities(target: str, scan_type: str = "basic", timeout: int = 60) -> str:
    """
    Perform basic security vulnerability scanning.
    
    Args:
        target: Target hostname or IP address
        scan_type: Type of scan (basic, service, vuln)
        timeout: Timeout in seconds (default: 60)
    """
    try:
        if not target:
            return json.dumps({
                "type": "error",
                "message": "Target cannot be empty"
            })
        
        # Define scan commands based on type
        scan_commands = {
            "basic": ["nmap", "-sS", "-O", target],
            "service": ["nmap", "-sV", "-sC", target],
            "vuln": ["nmap", "--script", "vuln", target]
        }
        
        if scan_type not in scan_commands:
            return json.dumps({
                "type": "error",
                "message": f"Invalid scan type. Use: {', '.join(scan_commands.keys())}"
            })
        
        try:
            result = subprocess.run(
                scan_commands[scan_type],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return json.dumps({
                "type": "security_scan_result",
                "target": target,
                "scan_type": scan_type,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "message": f"Security scan ({scan_type}) of {target} completed"
            })
        except FileNotFoundError:
            return json.dumps({
                "type": "error",
                "message": "nmap not available for security scanning"
            })
        
    except subprocess.TimeoutExpired:
        return json.dumps({
            "type": "error",
            "message": f"Security scan of {target} timed out after {timeout} seconds"
        })
    except Exception as e:
        return json.dumps({
            "type": "error",
            "message": f"Error performing security scan: {str(e)}"
        })

@mcp.tool()
async def run_command(command: str, timeout: int = 10) -> str:
    """
    Run a safe system command with timeout.
    
    Args:
        command: Command to run (limited to safe commands)
        timeout: Timeout in seconds (default: 10)
    """
    try:
        # Whitelist of safe commands including networking and security tools
        safe_commands = [
            "ls", "pwd", "date", "whoami", "uname", "df", "free",
            "ps", "top", "htop", "nvidia-smi", "jetson_release",
            "python3", "pip3", "ollama", "ping", "traceroute", "nslookup",
            "dig", "netstat", "ss", "lsof", "arp", "route", "ip",
            "ifconfig", "iwconfig", "curl", "wget", "nmap", "nc",
            "tcpdump", "tshark", "wireshark", "iptables", "ufw",
            "host", "whois", "telnet", "ssh", "scp", "rsync"
        ]
        
        # Check if command starts with a safe command
        command_parts = command.split()
        if not command_parts or command_parts[0] not in safe_commands:
            return json.dumps({
                "type": "error",
                "message": f"Command not allowed. Safe commands: {', '.join(safe_commands)}"
            })
        
        # Run command with timeout
        result = subprocess.run(
            command.split(),
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        return json.dumps({
            "type": "command_result",
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "message": f"Command '{command}' executed successfully"
        })
        
    except subprocess.TimeoutExpired:
        return json.dumps({
            "type": "error",
            "message": f"Command '{command}' timed out after {timeout} seconds"
        })
    except Exception as e:
        return json.dumps({
            "type": "error",
            "message": f"Error running command: {str(e)}"
        })

if __name__ == "__main__":
    print("Starting Simple MCP Server for Edge AI...", file=sys.stderr)
    print(f"Available tools: {len(mcp.tools)} tools registered", file=sys.stderr)
    
    # Run the MCP server
    mcp.run(transport='stdio')
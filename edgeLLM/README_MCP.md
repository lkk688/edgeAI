# Ollama MCP Gradio Client

A Model Context Protocol (MCP) client built with Gradio, optimized for **NVIDIA Jetson Orin Nano** and **Apple Silicon** devices. This client uses local Ollama models instead of cloud APIs, making it perfect for edge AI applications.

Referenced and modified based on [Gradio's MCP example](https://www.gradio.app/main/guides/building-an-mcp-client-with-gradio), which makes call to Claude API. We have modified the code to make calls to local Ollama models instead and optimized for edge devices.

## Features

### 🤖 Local AI Integration
- **Ollama Support**: Direct integration with local Ollama models
- **No API Keys**: Completely offline operation
- **Model Flexibility**: Support for any Ollama-compatible model
- **Streaming Responses**: Real-time response generation

### 🔧 MCP Integration
- **Tool Calling**: LLMs can execute tools through MCP protocol
- **Built-in MCP Server**: Simple server with essential tools
- **External MCP Support**: Connect to any MCP-compatible server
- **Tool Discovery**: Automatic tool detection and listing

### ⚡ Edge Optimization
- **Jetson Orin Nano**: CUDA memory management, power efficiency
- **Apple Silicon**: MPS acceleration, unified memory optimization
- **Resource Monitoring**: Real-time system metrics
- **Adaptive Performance**: Automatic optimization based on hardware

### 📊 Real-time Monitoring
- **System Metrics**: CPU, memory, GPU utilization
- **Platform Detection**: Automatic hardware identification
- **Live Updates**: Real-time performance dashboard
- **Resource Alerts**: Warnings for resource constraints

### 💾 Chat Management
- **Export Functionality**: Save conversations in multiple formats
- **Session Persistence**: Maintain chat history
- **Model Switching**: Change models mid-conversation
- **Clear History**: Reset conversations as needed

### 🛠️ Tool Integration
- **Image Generation**: ASCII art and text-based image descriptions
- **System Information**: Detailed hardware and software info
- **File Operations**: Create, read, and manage files
- **Command Execution**: Safe command execution with whitelist

### 🌐 Networking & Cybersecurity Tools

- **Network Diagnostics**: Ping, traceroute, port scanning
- **HTTP Operations**: GET/POST requests with custom headers
- **Protocol Testing**: TCP/UDP packet sending and receiving
- **DNS Operations**: Comprehensive DNS lookup and resolution
- **Packet Analysis**: Network packet capture and inspection
- **Security Scanning**: Vulnerability assessment and port analysis
- **Interface Monitoring**: Network interface information and statistics

Network Diagnostics:

- `network_ping` - Test network connectivity with customizable packet count and timeout
- `network_traceroute` - Trace network routes with hop limit control
- `network_interface_info` - Get detailed network interface information
Port Scanning & Security:

- `network_scan_ports` - Scan ports using nmap or socket connections
- `security_scan_vulnerabilities` - Perform security vulnerability assessments
- `packet_capture` - Capture network packets with filtering
Protocol Communication:

- `http_request` - Send HTTP requests (GET/POST/PUT/DELETE) with custom headers
- `send_udp_packet` - Send UDP packets and receive responses
- `tcp_connect` - Establish TCP connections with data exchange

DNS Operations:

- `dns_lookup` - Perform DNS lookups for various record types (A, MX, NS, TXT, etc.)

The Gradio MCP client can now handle requests like:

- "Can you ping google.com and check connectivity?"
- "Scan ports 22, 80, 443 on localhost"
- "Send a GET request to httpbin.org/get"
- "Look up the MX records for example.com"
- "Perform a basic security scan of 192.168.1.1"
- "Capture 10 network packets on any interface"

- Install Dependencies : pip install -r requirements_mcp.txt
- Install System Tools : sudo apt install nmap tcpdump wireshark-common (Linux) or brew install nmap tcpdump (macOS)
- Test Functionality : Run python test_networking_standalone.py
- Start MCP Server : python simple_mcp_server.py
- Launch Gradio Client : python ollama_mcp_gradio_client.py

## Prerequisites

### System Requirements

- **NVIDIA Jetson Orin Nano** OR **Apple Silicon Mac** (M1/M2/M3)
- Python 3.8+
- Ollama installed and running
- At least 8GB RAM (16GB recommended)

### Software Dependencies

1. **Install Ollama**:
   ```bash
   # On macOS
   brew install ollama
   
   # On Linux (including Jetson)
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Install Python Dependencies**:
   ```bash
   pip install -r requirements_mcp.txt
   ```

3. **Pull an Ollama Model**:
   ```bash
   # Lightweight models for edge devices
   ollama pull llama3.2:1b    # 1.3GB - Very fast
   ollama pull llama3.2:3b    # 2.0GB - Good balance
   ollama pull qwen2.5:3b     # 2.0GB - Good for coding
   
   # Larger models (if you have enough RAM)
   ollama pull llama3.2:8b    # 4.7GB - Better quality
   ```

## Quick Start

### 1. Start Ollama Server
```bash
ollama serve
```

### 2. Run the MCP Client
```bash
python3 ollama_mcp_gradio_client.py
```

### 3. Access the Web Interface
Open your browser and go to: `http://localhost:7861`

## Using MCP Servers

### Built-in Simple MCP Server

The repository includes a simple MCP server optimized for edge devices:

```bash
# In the Gradio interface, connect to:
./simple_mcp_server.py
```

#### 🎨 Creative Tools
- `generate_simple_image`: Create ASCII art or image descriptions
- `get_system_info`: Get detailed system information

#### 📁 File Operations
- `create_file`: Create text files
- `read_file`: Read file contents
- `list_directory`: List directory contents

#### 💻 System Tools
- `run_command`: Execute safe system commands

#### 🌐 Networking Tools
- `network_ping`: Test network connectivity to hosts
- `network_traceroute`: Trace network routes and hops
- `network_scan_ports`: Scan ports using nmap or socket connections
- `network_interface_info`: Get network interface information
- `dns_lookup`: Perform DNS lookups for various record types

#### 🔗 Protocol Tools
- `http_request`: Send HTTP GET/POST/PUT/DELETE requests
- `send_udp_packet`: Send UDP packets to hosts
- `tcp_connect`: Establish TCP connections and send data

#### 🔍 Security Tools
- `packet_capture`: Capture network packets using tcpdump/tshark
- `security_scan_vulnerabilities`: Perform security vulnerability scans

#### Starting the Simple MCP Server

```bash
# Start the server on default port 8000
python simple_mcp_server.py

# Start on custom port
python simple_mcp_server.py --port 8080
```

#### Testing Networking Tools

```bash
# Run comprehensive tests for all networking tools
python test_mcp_networking_tools.py
```

**Note**: Some networking tools require:
- Root/administrator privileges (packet capture, some scans)
- Network tools installed (nmap, tcpdump, wireshark)
- Network connectivity for external tests

### External MCP Servers

You can also connect to external MCP servers. For example, to use image generation:

1. **Create an image generation MCP server** (based on the Gradio tutorial):
   ```python
   # Save as image_mcp_server.py
   from mcp.server.fastmcp import FastMCP
   from gradio_client import Client
   import json
   
   mcp = FastMCP("image_generator")
   
   @mcp.tool()
   async def generate_image(prompt: str, width: int = 512, height: int = 512) -> str:
       client = Client("https://stabilityai-stable-diffusion-xl.hf.space/")
       result = client.predict(prompt, api_name="/predict")
       return json.dumps({"type": "image", "url": result})
   
   if __name__ == "__main__":
       mcp.run(transport='stdio')
   ```

2. **Connect in the Gradio interface**:
   - Enter `./image_mcp_server.py` in the "Server Path" field
   - Click "Connect"
   - Enable "MCP Tools" checkbox
   - Ask for image generation: "Generate an image of a sunset"

## Platform-Specific Optimizations

### NVIDIA Jetson Orin Nano

- **Memory Management**: Automatically limits CUDA memory usage to 70%
- **Power Efficiency**: Optimized for Jetson's power constraints
- **GPU Monitoring**: Uses `jetson-stats` if available
- **Thermal Management**: Monitors temperature and throttling

**Recommended Settings:**
```bash
# Set power mode for better performance
sudo nvpmodel -m 0  # MAXN mode
sudo jetson_clocks   # Max clocks

# Monitor with jetson-stats
sudo pip install jetson-stats
jtop
```

### Apple Silicon (M1/M2/M3)

- **MPS Support**: Utilizes Metal Performance Shaders when available
- **Unified Memory**: Optimized for Apple's unified memory architecture
- **Power Efficiency**: Leverages Apple Silicon's efficiency cores
- **Native Performance**: Runs natively on ARM64

**Recommended Settings:**
```bash
# Install Ollama with Apple Silicon optimization
brew install ollama

# Use models optimized for Apple Silicon
ollama pull llama3.2:3b
```

## Configuration

### Environment Variables

```bash
# Disable Gradio analytics (recommended for edge devices)
export GRADIO_ANALYTICS_ENABLED=False

# For Jetson devices
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# For Apple Silicon
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Memory Optimization

**For Jetson (8GB RAM):**
- Use 1B or 3B parameter models
- Limit CUDA memory fraction to 0.7
- Close unnecessary applications

**For Apple Silicon (8GB+ unified memory):**
- 3B models work well on 8GB
- 8B models recommended for 16GB+
- MPS automatically manages memory

## Security Considerations

### ⚠️ Important Security Notes

The networking and cybersecurity tools included in this MCP server are powerful and should be used responsibly:

#### Network Scanning Ethics
- **Only scan systems you own or have explicit permission to test**
- Unauthorized network scanning may violate laws and terms of service
- Always follow responsible disclosure practices
- Be aware of your organization's security policies

#### Tool Privileges
- **Packet capture requires root/administrator privileges**
- Some security scans may trigger security alerts
- Network tools may be blocked by firewalls
- Consider running in isolated environments for testing

#### Data Privacy
- Network captures may contain sensitive information
- HTTP requests may expose credentials or personal data
- Be cautious when sharing tool outputs
- Consider data retention and disposal policies

#### Safe Usage Guidelines
1. **Test in controlled environments first**
2. **Use minimal necessary privileges**
3. **Monitor and log tool usage**
4. **Regularly update security tools**
5. **Follow your organization's cybersecurity policies**

## Troubleshooting

### Common Issues

1. **"Ollama server not running"**
   ```bash
   ollama serve
   # Wait a few seconds, then try again
   ```

2. **"No models available"**
   ```bash
   ollama pull llama3.2:3b
   ollama list  # Verify model is installed
   ```

3. **"MCP server connection failed"**
   - Check the server path is correct
   - Ensure the MCP server file is executable
   - Check Python dependencies are installed

4. **High memory usage on Jetson**
   ```bash
   # Use smaller models
   ollama pull llama3.2:1b
   
   # Monitor memory
   free -h
   jtop  # If jetson-stats installed
   ```

5. **Slow performance**
   - Use smaller models (1B-3B parameters)
   - Close other applications
   - Check system temperature (Jetson)
   - Ensure adequate cooling

6. **Networking tools not working**
   - Install required tools: `sudo apt install nmap tcpdump wireshark-common`
   - Check user privileges for packet capture
   - Verify network connectivity
   - Check firewall rules

## Usage Examples

### Basic Chat
1. Start Ollama and pull a model
2. Launch the Gradio client
3. Select your model and start chatting

### Using MCP Tools
1. Connect to an MCP server
2. Ask the LLM to use tools: "Can you create an ASCII art of a cat?"
3. The LLM will automatically call the appropriate MCP tool

### Networking & Cybersecurity Tasks

#### Network Diagnostics
```
User: "Can you ping google.com and check if it's reachable?"
LLM: Uses network_ping tool to test connectivity

User: "Trace the route to cloudflare.com"
LLM: Uses network_traceroute to show network path

User: "Check what network interfaces are available on this system"
LLM: Uses network_interface_info to display interface details
```

#### Port Scanning & Security
```
User: "Scan ports 22, 80, 443 on localhost"
LLM: Uses network_scan_ports to check port status

User: "Perform a basic security scan of 192.168.1.1"
LLM: Uses security_scan_vulnerabilities for assessment

User: "Capture 10 network packets on any interface"
LLM: Uses packet_capture to monitor network traffic
```

#### HTTP Operations
```
User: "Send a GET request to httpbin.org/get"
LLM: Uses http_request to make HTTP calls

User: "POST some JSON data to an API endpoint"
LLM: Uses http_request with POST method and custom headers
```

#### DNS & Protocol Testing
```
User: "Look up the MX records for example.com"
LLM: Uses dns_lookup to query DNS records

User: "Send a UDP packet to 8.8.8.8 port 53"
LLM: Uses send_udp_packet for protocol testing

User: "Test TCP connection to google.com port 80"
LLM: Uses tcp_connect to establish connections
```

### System Monitoring
- View real-time system metrics in the sidebar
- Monitor GPU usage during inference
- Track memory consumption

### Chat Export
- Use the "Export Chat" button to save conversations
- Choose between Markdown and JSON formats

## Development

### Adding Custom MCP Tools

1. **Create a new MCP server**:
   ```python
   from mcp.server.fastmcp import FastMCP
   
   mcp = FastMCP("my_custom_server")
   
   @mcp.tool()
   async def my_custom_tool(param: str) -> str:
       # Your tool logic here
       return json.dumps({"result": "success"})
   
   if __name__ == "__main__":
       mcp.run(transport='stdio')
   ```

2. **Connect in the client**: Enter the path to your server file

3. **Use the tool**: Enable MCP tools and mention your tool in chat

### Extending the Client

The client is modular and can be extended:

- **Add new backends**: Modify the `process_with_ollama` function
- **Custom UI components**: Add new Gradio components
- **Enhanced monitoring**: Extend the system monitoring utilities

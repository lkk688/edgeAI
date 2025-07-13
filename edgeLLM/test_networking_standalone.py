#!/usr/bin/env python3
"""
Standalone Test Script for Networking Tools

This script demonstrates the networking functionality without requiring MCP setup.
It tests the core networking functions that would be used in the MCP server.

Usage:
    python test_networking_standalone.py
"""

import asyncio
import json
import subprocess
import socket
import time
from typing import Dict, Any

# Check for optional networking libraries
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️  requests library not available - HTTP tests will be skipped")

class NetworkingDemo:
    """Demonstration of networking tools functionality"""
    
    def __init__(self):
        self.test_results = []
        self.test_hosts = {
            "local": "127.0.0.1",
            "google_dns": "8.8.8.8",
            "cloudflare_dns": "1.1.1.1",
            "example_domain": "example.com"
        }
    
    def log_result(self, test_name: str, success: bool, result: Any, error: str = None):
        """Log test results"""
        self.test_results.append({
            "test": test_name,
            "success": success,
            "result": result,
            "error": error
        })
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if error:
            print(f"   Error: {error}")
        elif success and isinstance(result, dict):
            print(f"   Result: {result.get('message', 'Success')}")
    
    async def demo_ping(self):
        """Demonstrate ping functionality"""
        print("\n🔍 Testing Ping Functionality...")
        
        try:
            # Test ping to localhost
            result = subprocess.run(
                ["ping", "-c", "2", self.test_hosts["local"]],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            ping_result = {
                "type": "ping_result",
                "host": self.test_hosts["local"],
                "success": result.returncode == 0,
                "message": f"Ping to localhost {'successful' if result.returncode == 0 else 'failed'}",
                "output_lines": len(result.stdout.split('\n')) if result.stdout else 0
            }
            
            self.log_result("ping_localhost", ping_result["success"], ping_result)
            
        except Exception as e:
            self.log_result("ping_localhost", False, None, str(e))
        
        # Test ping to external host
        try:
            result = subprocess.run(
                ["ping", "-c", "2", self.test_hosts["google_dns"]],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            ping_result = {
                "type": "ping_result",
                "host": self.test_hosts["google_dns"],
                "success": result.returncode == 0,
                "message": f"Ping to Google DNS {'successful' if result.returncode == 0 else 'failed'}"
            }
            
            self.log_result("ping_google_dns", ping_result["success"], ping_result)
            
        except Exception as e:
            self.log_result("ping_google_dns", False, None, str(e))
    
    async def demo_port_scan(self):
        """Demonstrate port scanning using socket connections"""
        print("\n🔍 Testing Port Scanning...")
        
        # Test common ports on localhost
        test_ports = [22, 80, 443, 8080]
        scan_results = []
        
        for port in test_ports:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((self.test_hosts["local"], port))
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
        
        port_scan_result = {
            "type": "port_scan_result",
            "host": self.test_hosts["local"],
            "method": "socket",
            "results": scan_results,
            "message": f"Port scan of localhost completed"
        }
        
        open_ports = [r["port"] for r in scan_results if r["status"] == "open"]
        success = len(scan_results) > 0
        
        self.log_result("port_scan_localhost", success, port_scan_result)
        if open_ports:
            print(f"   Open ports found: {open_ports}")
    
    async def demo_http_request(self):
        """Demonstrate HTTP request functionality"""
        print("\n🔍 Testing HTTP Requests...")
        
        if not REQUESTS_AVAILABLE:
            self.log_result("http_request_test", False, None, "requests library not available")
            return
        
        try:
            # Test GET request to httpbin
            response = requests.get("https://httpbin.org/get", timeout=15)
            
            http_result = {
                "type": "http_response",
                "url": "https://httpbin.org/get",
                "method": "GET",
                "status_code": response.status_code,
                "content_length": len(response.text),
                "message": f"HTTP GET request completed with status {response.status_code}"
            }
            
            success = response.status_code == 200
            self.log_result("http_get_request", success, http_result)
            
        except Exception as e:
            self.log_result("http_get_request", False, None, str(e))
        
        # Test POST request
        try:
            data = {"test": "data", "timestamp": time.time()}
            response = requests.post(
                "https://httpbin.org/post",
                json=data,
                timeout=15
            )
            
            http_result = {
                "type": "http_response",
                "url": "https://httpbin.org/post",
                "method": "POST",
                "status_code": response.status_code,
                "message": f"HTTP POST request completed with status {response.status_code}"
            }
            
            success = response.status_code == 200
            self.log_result("http_post_request", success, http_result)
            
        except Exception as e:
            self.log_result("http_post_request", False, None, str(e))
    
    async def demo_udp_communication(self):
        """Demonstrate UDP packet sending"""
        print("\n🔍 Testing UDP Communication...")
        
        try:
            # Send UDP packet to DNS server
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(5)
            
            message = "test packet"
            bytes_sent = sock.sendto(message.encode('utf-8'), (self.test_hosts["google_dns"], 53))
            
            # Try to receive response (DNS servers typically don't respond to random UDP)
            try:
                sock.settimeout(2)
                response, addr = sock.recvfrom(1024)
                response_received = True
            except socket.timeout:
                response_received = False
            
            sock.close()
            
            udp_result = {
                "type": "udp_send_result",
                "host": self.test_hosts["google_dns"],
                "port": 53,
                "bytes_sent": bytes_sent,
                "response_received": response_received,
                "message": f"UDP packet sent to Google DNS"
            }
            
            self.log_result("udp_send_dns", True, udp_result)
            
        except Exception as e:
            self.log_result("udp_send_dns", False, None, str(e))
    
    async def demo_tcp_connection(self):
        """Demonstrate TCP connection"""
        print("\n🔍 Testing TCP Connection...")
        
        try:
            # Connect to Google DNS port 53
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            
            start_time = time.time()
            sock.connect((self.test_hosts["google_dns"], 53))
            connect_time = time.time() - start_time
            
            sock.close()
            
            tcp_result = {
                "type": "tcp_connect_result",
                "host": self.test_hosts["google_dns"],
                "port": 53,
                "connect_time": round(connect_time * 1000, 2),  # ms
                "message": f"TCP connection to Google DNS successful"
            }
            
            self.log_result("tcp_connect_dns", True, tcp_result)
            
        except Exception as e:
            self.log_result("tcp_connect_dns", False, None, str(e))
    
    async def demo_dns_lookup(self):
        """Demonstrate DNS lookup using system commands"""
        print("\n🔍 Testing DNS Lookup...")
        
        # Try dig first, then nslookup
        for cmd in ["dig", "nslookup"]:
            try:
                if cmd == "dig":
                    result = subprocess.run(
                        ["dig", "+short", "-t", "A", self.test_hosts["example_domain"]],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                else:
                    result = subprocess.run(
                        ["nslookup", self.test_hosts["example_domain"]],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                
                dns_result = {
                    "type": "dns_lookup_result",
                    "hostname": self.test_hosts["example_domain"],
                    "command": cmd,
                    "success": result.returncode == 0,
                    "output_lines": len(result.stdout.split('\n')) if result.stdout else 0,
                    "message": f"DNS lookup using {cmd} {'successful' if result.returncode == 0 else 'failed'}"
                }
                
                self.log_result(f"dns_lookup_{cmd}", dns_result["success"], dns_result)
                
                if dns_result["success"]:
                    break  # If one works, don't try the other
                    
            except FileNotFoundError:
                self.log_result(f"dns_lookup_{cmd}", False, None, f"{cmd} command not found")
            except Exception as e:
                self.log_result(f"dns_lookup_{cmd}", False, None, str(e))
    
    async def demo_network_interface_info(self):
        """Demonstrate network interface information gathering"""
        print("\n🔍 Testing Network Interface Info...")
        
        # Try multiple commands
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
                    interface_result = {
                        "type": "network_interface_info",
                        "command": " ".join(cmd),
                        "output_lines": len(result.stdout.split('\n')),
                        "message": f"Network interface info retrieved using {cmd[0]}"
                    }
                    
                    self.log_result("network_interface_info", True, interface_result)
                    return
                    
            except FileNotFoundError:
                continue
            except Exception as e:
                self.log_result("network_interface_info", False, None, str(e))
                return
        
        self.log_result("network_interface_info", False, None, "No suitable network interface command available")
    
    async def run_all_demos(self):
        """Run all networking demonstrations"""
        print("🚀 Starting Networking Tools Demonstration")
        print("=" * 60)
        print("This demo shows the networking functionality that would be")
        print("available through the MCP server tools.")
        print("=" * 60)
        
        # Run all demonstrations
        await self.demo_ping()
        await self.demo_port_scan()
        await self.demo_http_request()
        await self.demo_udp_communication()
        await self.demo_tcp_connection()
        await self.demo_dns_lookup()
        await self.demo_network_interface_info()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print demonstration summary"""
        print("\n" + "=" * 60)
        print("📊 DEMONSTRATION SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Demonstrations: {total_tests}")
        print(f"✅ Successful: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ Failed Demonstrations:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"   - {result['test']}: {result.get('error', 'Unknown error')}")
        
        print("\n📝 Notes:")
        print("   - These functions demonstrate the core networking capabilities")
        print("   - In the MCP server, these would be available as LLM tools")
        print("   - Some failures are expected (closed ports, missing tools)")
        print("   - Network connectivity required for external tests")
        
        print("\n🔧 Available MCP Tools:")
        tools = [
            "network_ping", "network_traceroute", "network_scan_ports",
            "http_request", "send_udp_packet", "tcp_connect",
            "network_interface_info", "dns_lookup", "packet_capture",
            "security_scan_vulnerabilities"
        ]
        for tool in tools:
            print(f"   - {tool}")

def main():
    """Main demonstration function"""
    print("Networking Tools Demonstration")
    print("This script demonstrates the networking functionality available in the MCP server.")
    print("\n⚠️  Note: This is a demonstration of core functionality.")
    print("⚠️  The full MCP server includes additional security and validation.")
    
    # Run demonstrations
    demo = NetworkingDemo()
    asyncio.run(demo.run_all_demos())

if __name__ == "__main__":
    main()
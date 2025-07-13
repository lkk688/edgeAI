#!/usr/bin/env python3
"""
Test script for MCP Networking and Cybersecurity Tools

This script tests all the networking and cybersecurity tools added to the simple_mcp_server.py.
It includes comprehensive tests for each tool with various scenarios.

Usage:
    python test_mcp_networking_tools.py

Requirements:
    - Python 3.7+
    - mcp library
    - requests library
    - Network connectivity
    - Some tools require root privileges (tcpdump, nmap)
"""

import asyncio
import json
import sys
import time
from typing import Dict, Any

# Import the MCP server functions
try:
    from simple_mcp_server import (
        network_ping,
        network_traceroute,
        network_scan_ports,
        http_request,
        send_udp_packet,
        tcp_connect,
        network_interface_info,
        dns_lookup,
        packet_capture,
        security_scan_vulnerabilities
    )
except ImportError as e:
    print(f"Error importing MCP server functions: {e}")
    print("Make sure simple_mcp_server.py is in the same directory")
    sys.exit(1)

class NetworkingToolsTester:
    """Test suite for networking and cybersecurity MCP tools"""
    
    def __init__(self):
        self.test_results = []
        self.test_hosts = {
            "local": "127.0.0.1",
            "google_dns": "8.8.8.8",
            "cloudflare_dns": "1.1.1.1",
            "example_domain": "example.com",
            "httpbin": "httpbin.org"
        }
    
    def log_test(self, test_name: str, success: bool, result: Any, error: str = None):
        """Log test results"""
        self.test_results.append({
            "test": test_name,
            "success": success,
            "result": result,
            "error": error,
            "timestamp": time.time()
        })
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if error:
            print(f"   Error: {error}")
    
    async def test_network_ping(self):
        """Test network ping functionality"""
        print("\n🔍 Testing Network Ping...")
        
        # Test 1: Ping localhost
        try:
            result = await network_ping(self.test_hosts["local"], count=2, timeout=5)
            result_data = json.loads(result)
            success = result_data.get("type") == "ping_result" and result_data.get("success", False)
            self.log_test("ping_localhost", success, result_data)
        except Exception as e:
            self.log_test("ping_localhost", False, None, str(e))
        
        # Test 2: Ping Google DNS
        try:
            result = await network_ping(self.test_hosts["google_dns"], count=2, timeout=10)
            result_data = json.loads(result)
            success = result_data.get("type") == "ping_result"
            self.log_test("ping_google_dns", success, result_data)
        except Exception as e:
            self.log_test("ping_google_dns", False, None, str(e))
        
        # Test 3: Ping invalid host
        try:
            result = await network_ping("invalid.nonexistent.domain", count=1, timeout=5)
            result_data = json.loads(result)
            success = result_data.get("type") in ["ping_result", "error"]
            self.log_test("ping_invalid_host", success, result_data)
        except Exception as e:
            self.log_test("ping_invalid_host", False, None, str(e))
    
    async def test_network_traceroute(self):
        """Test network traceroute functionality"""
        print("\n🔍 Testing Network Traceroute...")
        
        # Test 1: Traceroute to Google DNS
        try:
            result = await network_traceroute(self.test_hosts["google_dns"], max_hops=10, timeout=20)
            result_data = json.loads(result)
            success = result_data.get("type") in ["traceroute_result", "error"]
            self.log_test("traceroute_google_dns", success, result_data)
        except Exception as e:
            self.log_test("traceroute_google_dns", False, None, str(e))
        
        # Test 2: Traceroute to localhost
        try:
            result = await network_traceroute(self.test_hosts["local"], max_hops=5, timeout=10)
            result_data = json.loads(result)
            success = result_data.get("type") in ["traceroute_result", "error"]
            self.log_test("traceroute_localhost", success, result_data)
        except Exception as e:
            self.log_test("traceroute_localhost", False, None, str(e))
    
    async def test_port_scanning(self):
        """Test port scanning functionality"""
        print("\n🔍 Testing Port Scanning...")
        
        # Test 1: Scan common ports on localhost
        try:
            result = await network_scan_ports(self.test_hosts["local"], "22,80,443,8080", timeout=15)
            result_data = json.loads(result)
            success = result_data.get("type") == "port_scan_result"
            self.log_test("port_scan_localhost", success, result_data)
        except Exception as e:
            self.log_test("port_scan_localhost", False, None, str(e))
        
        # Test 2: Scan Google DNS (port 53)
        try:
            result = await network_scan_ports(self.test_hosts["google_dns"], "53", timeout=10)
            result_data = json.loads(result)
            success = result_data.get("type") == "port_scan_result"
            self.log_test("port_scan_dns", success, result_data)
        except Exception as e:
            self.log_test("port_scan_dns", False, None, str(e))
        
        # Test 3: Invalid port range
        try:
            result = await network_scan_ports(self.test_hosts["local"], "99999", timeout=5)
            result_data = json.loads(result)
            success = result_data.get("type") == "error"
            self.log_test("port_scan_invalid", success, result_data)
        except Exception as e:
            self.log_test("port_scan_invalid", False, None, str(e))
    
    async def test_http_requests(self):
        """Test HTTP request functionality"""
        print("\n🔍 Testing HTTP Requests...")
        
        # Test 1: GET request to httpbin
        try:
            result = await http_request(f"https://{self.test_hosts['httpbin']}/get", "GET", timeout=15)
            result_data = json.loads(result)
            success = result_data.get("type") == "http_response" and result_data.get("status_code") == 200
            self.log_test("http_get_request", success, result_data)
        except Exception as e:
            self.log_test("http_get_request", False, None, str(e))
        
        # Test 2: POST request with data
        try:
            headers = json.dumps({"Content-Type": "application/json"})
            data = json.dumps({"test": "data"})
            result = await http_request(f"https://{self.test_hosts['httpbin']}/post", "POST", headers, data, timeout=15)
            result_data = json.loads(result)
            success = result_data.get("type") == "http_response" and result_data.get("status_code") == 200
            self.log_test("http_post_request", success, result_data)
        except Exception as e:
            self.log_test("http_post_request", False, None, str(e))
        
        # Test 3: Invalid URL
        try:
            result = await http_request("invalid-url", "GET", timeout=5)
            result_data = json.loads(result)
            success = result_data.get("type") == "error"
            self.log_test("http_invalid_url", success, result_data)
        except Exception as e:
            self.log_test("http_invalid_url", False, None, str(e))
    
    async def test_udp_communication(self):
        """Test UDP packet sending"""
        print("\n🔍 Testing UDP Communication...")
        
        # Test 1: Send UDP packet to DNS server
        try:
            result = await send_udp_packet(self.test_hosts["google_dns"], 53, "test", timeout=5)
            result_data = json.loads(result)
            success = result_data.get("type") == "udp_send_result"
            self.log_test("udp_send_dns", success, result_data)
        except Exception as e:
            self.log_test("udp_send_dns", False, None, str(e))
        
        # Test 2: Send UDP to localhost (likely no response)
        try:
            result = await send_udp_packet(self.test_hosts["local"], 12345, "hello", timeout=3)
            result_data = json.loads(result)
            success = result_data.get("type") == "udp_send_result"
            self.log_test("udp_send_localhost", success, result_data)
        except Exception as e:
            self.log_test("udp_send_localhost", False, None, str(e))
        
        # Test 3: Invalid port
        try:
            result = await send_udp_packet(self.test_hosts["local"], 99999, "test", timeout=3)
            result_data = json.loads(result)
            success = result_data.get("type") == "error"
            self.log_test("udp_invalid_port", success, result_data)
        except Exception as e:
            self.log_test("udp_invalid_port", False, None, str(e))
    
    async def test_tcp_connection(self):
        """Test TCP connection functionality"""
        print("\n🔍 Testing TCP Connection...")
        
        # Test 1: Connect to Google DNS port 53
        try:
            result = await tcp_connect(self.test_hosts["google_dns"], 53, timeout=10)
            result_data = json.loads(result)
            success = result_data.get("type") == "tcp_connect_result"
            self.log_test("tcp_connect_dns", success, result_data)
        except Exception as e:
            self.log_test("tcp_connect_dns", False, None, str(e))
        
        # Test 2: Connect to HTTP port with message
        try:
            result = await tcp_connect("httpbin.org", 80, "GET / HTTP/1.1\r\nHost: httpbin.org\r\n\r\n", timeout=10)
            result_data = json.loads(result)
            success = result_data.get("type") == "tcp_connect_result"
            self.log_test("tcp_connect_http", success, result_data)
        except Exception as e:
            self.log_test("tcp_connect_http", False, None, str(e))
        
        # Test 3: Connect to closed port
        try:
            result = await tcp_connect(self.test_hosts["local"], 12345, timeout=3)
            result_data = json.loads(result)
            success = result_data.get("type") in ["tcp_connect_result", "error"]
            self.log_test("tcp_connect_closed_port", success, result_data)
        except Exception as e:
            self.log_test("tcp_connect_closed_port", False, None, str(e))
    
    async def test_network_interface_info(self):
        """Test network interface information"""
        print("\n🔍 Testing Network Interface Info...")
        
        try:
            result = await network_interface_info()
            result_data = json.loads(result)
            success = result_data.get("type") in ["network_interface_info", "error"]
            self.log_test("network_interface_info", success, result_data)
        except Exception as e:
            self.log_test("network_interface_info", False, None, str(e))
    
    async def test_dns_lookup(self):
        """Test DNS lookup functionality"""
        print("\n🔍 Testing DNS Lookup...")
        
        # Test 1: A record lookup
        try:
            result = await dns_lookup(self.test_hosts["example_domain"], "A", timeout=10)
            result_data = json.loads(result)
            success = result_data.get("type") in ["dns_lookup_result", "error"]
            self.log_test("dns_lookup_a_record", success, result_data)
        except Exception as e:
            self.log_test("dns_lookup_a_record", False, None, str(e))
        
        # Test 2: MX record lookup
        try:
            result = await dns_lookup(self.test_hosts["example_domain"], "MX", timeout=10)
            result_data = json.loads(result)
            success = result_data.get("type") in ["dns_lookup_result", "error"]
            self.log_test("dns_lookup_mx_record", success, result_data)
        except Exception as e:
            self.log_test("dns_lookup_mx_record", False, None, str(e))
        
        # Test 3: Invalid hostname
        try:
            result = await dns_lookup("invalid.nonexistent.domain.xyz", "A", timeout=5)
            result_data = json.loads(result)
            success = result_data.get("type") in ["dns_lookup_result", "error"]
            self.log_test("dns_lookup_invalid", success, result_data)
        except Exception as e:
            self.log_test("dns_lookup_invalid", False, None, str(e))
    
    async def test_packet_capture(self):
        """Test packet capture functionality (requires privileges)"""
        print("\n🔍 Testing Packet Capture...")
        
        # Note: This test might fail without root privileges
        try:
            result = await packet_capture("any", count=2, timeout=10)
            result_data = json.loads(result)
            success = result_data.get("type") in ["packet_capture_result", "error"]
            self.log_test("packet_capture_basic", success, result_data)
        except Exception as e:
            self.log_test("packet_capture_basic", False, None, str(e))
        
        # Test with filter
        try:
            result = await packet_capture("any", count=1, filter_expr="icmp", timeout=5)
            result_data = json.loads(result)
            success = result_data.get("type") in ["packet_capture_result", "error"]
            self.log_test("packet_capture_filtered", success, result_data)
        except Exception as e:
            self.log_test("packet_capture_filtered", False, None, str(e))
    
    async def test_security_scanning(self):
        """Test security vulnerability scanning"""
        print("\n🔍 Testing Security Scanning...")
        
        # Test 1: Basic scan of localhost
        try:
            result = await security_scan_vulnerabilities(self.test_hosts["local"], "basic", timeout=30)
            result_data = json.loads(result)
            success = result_data.get("type") in ["security_scan_result", "error"]
            self.log_test("security_scan_basic", success, result_data)
        except Exception as e:
            self.log_test("security_scan_basic", False, None, str(e))
        
        # Test 2: Service scan
        try:
            result = await security_scan_vulnerabilities(self.test_hosts["google_dns"], "service", timeout=20)
            result_data = json.loads(result)
            success = result_data.get("type") in ["security_scan_result", "error"]
            self.log_test("security_scan_service", success, result_data)
        except Exception as e:
            self.log_test("security_scan_service", False, None, str(e))
        
        # Test 3: Invalid scan type
        try:
            result = await security_scan_vulnerabilities(self.test_hosts["local"], "invalid", timeout=5)
            result_data = json.loads(result)
            success = result_data.get("type") == "error"
            self.log_test("security_scan_invalid_type", success, result_data)
        except Exception as e:
            self.log_test("security_scan_invalid_type", False, None, str(e))
    
    async def run_all_tests(self):
        """Run all networking and cybersecurity tests"""
        print("🚀 Starting MCP Networking and Cybersecurity Tools Test Suite")
        print("=" * 60)
        
        # Run all test categories
        await self.test_network_ping()
        await self.test_network_traceroute()
        await self.test_port_scanning()
        await self.test_http_requests()
        await self.test_udp_communication()
        await self.test_tcp_connection()
        await self.test_network_interface_info()
        await self.test_dns_lookup()
        await self.test_packet_capture()
        await self.test_security_scanning()
        
        # Print summary
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print test results summary"""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ Failed Tests:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"   - {result['test']}: {result.get('error', 'Unknown error')}")
        
        print("\n📝 Notes:")
        print("   - Some tests may fail due to network connectivity issues")
        print("   - Packet capture and some security scans require root privileges")
        print("   - nmap, tcpdump, and other tools must be installed for full functionality")
        print("   - Firewall settings may affect some test results")

def main():
    """Main test function"""
    print("MCP Networking and Cybersecurity Tools Test Suite")
    print("This script tests all networking tools in simple_mcp_server.py")
    print("\n⚠️  Warning: Some tests require root privileges and network access")
    print("⚠️  Warning: Security scans should only be run on systems you own")
    
    # Ask for confirmation
    response = input("\nDo you want to proceed with the tests? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Tests cancelled.")
        return
    
    # Run tests
    tester = NetworkingToolsTester()
    asyncio.run(tester.run_all_tests())

if __name__ == "__main__":
    main()
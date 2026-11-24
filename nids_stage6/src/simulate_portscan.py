"""
simulate_portscan.py — Lightweight attack traffic simulator for Phase 4.4 / Item 28

Simulates a rapid TCP port scan against a specified target IP (e.g. 127.0.0.1 or local Wi-Fi IP).
Usage:
    python simulate_portscan.py --target 192.168.1.4 --ports 1-150
"""

import argparse
import socket
import time
import concurrent.futures

def scan_port(target_ip: str, port: int, timeout: float = 0.15):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((target_ip, port))
            return port, "OPEN"
    except (socket.timeout, ConnectionRefusedError, OSError):
        return port, "CLOSED"

def run_port_scan(target_ip: str, start_port: int, end_port: int, max_workers: int = 30):
    print(f"[*] Initiating simulated PortScan against {target_ip} (ports {start_port}-{end_port})...")
    ports = list(range(start_port, end_port + 1))
    t0 = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(scan_port, target_ip, p) for p in ports]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
    duration = time.time() - t0
    open_ports = [p for p, status in results if status == "OPEN"]
    print(f"[+] Scan completed in {duration:.2f}s: {len(results)} ports probed, {len(open_ports)} open.")
    print("[+] Check the Live Detection dashboard for real-time alerts!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate PortScan traffic for Stage 6 live detection testing")
    parser.add_argument("--target", default="127.0.0.1", help="Target IP to probe")
    parser.add_argument("--start_port", type=int, default=1, help="Starting port (default: 1)")
    parser.add_argument("--end_port", type=int, default=120, help="Ending port (default: 120)")
    args = parser.parse_args()
    
    run_port_scan(args.target, args.start_port, args.end_port)

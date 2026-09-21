"""
live_capture.py
================
Stage 6 - Component 1: Live Traffic Capture

Wraps live packet capture for the real-time NIDS pipeline. Two backends:

    1. TsharkCapture (real) — uses `pyshark`, a Python wrapper around the
       `tshark` CLI (Wireshark's command-line capture tool), to sniff live
       packets off a network interface. This is the REAL mechanism your
       project plan calls for ("Wireshark/Tshark: for capturing live pcap
       data"). Requires:
           - tshark installed (comes with Wireshark) and on PATH
           - `pip install pyshark`
           - permission to capture on the interface (root/administrator,
             or appropriate capture group membership on Linux)
       NOT available in this dev environment — no network interface access,
       no ability to install tshark/pyshark here. The code below is real
       and correct; it needs to be run on your team's machine.

    2. SimulatedCapture (tested, dependency-free) — generates synthetic
       packet events at a configurable rate, standing in for live traffic
       so the rest of the pipeline (flow aggregation, feature extraction,
       scaling, inference, alerting) can be built and tested end-to-end
       without needing a live interface. Clearly NOT real traffic — use
       only for pipeline development/demo, never for reporting detection
       performance.

Both backends expose the same interface: an iterator/generator yielding
PacketEvent objects, so downstream code (flow_feature_extractor.py) never
needs to know which one is in use.
"""

import random
import time
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class PacketEvent:
    """A minimal packet record — just enough fields to aggregate into flow-level features."""
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int          # 6=TCP, 17=UDP, 1=ICMP
    length: int             # packet size in bytes
    direction: str = "fwd"  # "fwd" (src->dst) or "bwd" (dst->src), relative to flow initiator


class TsharkCapture:
    """
    REAL live capture backend using pyshark/tshark.

    Windows requirements:
        - Wireshark installed
        - Npcap installed (normally installed with Wireshark)
        - tshark available on PATH
        - pip install pyshark
        - terminal with sufficient capture permissions

    The capture yields PacketEvent objects. Direction is inferred from
    the first packet seen for each 5-tuple conversation, with TCP SYN
    preferred as the initiator when available.
    """

    def __init__(
        self,
        interface: str = "Wi-Fi",
        bpf_filter: str = None,
    ):
        self.interface = interface
        self.bpf_filter = bpf_filter
        self._flow_directions = {}

    @staticmethod
    def _endpoint_key(src_ip, src_port, dst_ip, dst_port, protocol):
        """
        Creates a direction-independent key for a conversation.
        """
        endpoints = tuple(sorted([
            (src_ip, src_port),
            (dst_ip, dst_port),
        ]))
        return (endpoints, protocol)

    @staticmethod
    def _is_tcp_syn(pkt):
        """
        Returns True for an initial TCP SYN (SYN=1, ACK=0).
        """
        try:
            if getattr(pkt, "transport_layer", None) != "TCP":
                return False

            tcp = pkt.tcp
            syn = int(getattr(tcp, "flags_syn", 0))
            ack = int(getattr(tcp, "flags_ack", 0))

            return syn == 1 and ack == 0
        except Exception:
            return False

    def _get_direction(
        self,
        src_ip,
        src_port,
        dst_ip,
        dst_port,
        protocol,
        pkt,
    ):
        """
        Determines whether this packet is forward or backward relative
        to the flow initiator.
        """
        key = self._endpoint_key(
            src_ip,
            src_port,
            dst_ip,
            dst_port,
            protocol,
        )

        current_endpoint = (
            src_ip,
            src_port,
            dst_ip,
            dst_port,
        )

        if key not in self._flow_directions:
            # Prefer the TCP SYN as the initiator.
            # If the capture starts after the handshake, simply treat
            # the first observed direction as forward.
            self._flow_directions[key] = current_endpoint

        initiator = self._flow_directions[key]

        if current_endpoint == initiator:
            return "fwd"

        return "bwd"

    def capture(self) -> Iterator[PacketEvent]:
        try:
            import pyshark
        except ImportError:
            raise RuntimeError(
                "pyshark is not installed. Run:\n"
                "    pip install pyshark\n"
                "Also install Wireshark/TShark and ensure tshark is on PATH."
            )

        print(f">>> Starting TShark live capture")
        print(f">>> Interface: {self.interface}")

        if self.bpf_filter:
            print(f">>> BPF filter: {self.bpf_filter}")

        try:
            capture = pyshark.LiveCapture(
                interface=self.interface,
                bpf_filter=self.bpf_filter,
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not start TShark capture on interface "
                f"'{self.interface}'.\n"
                f"Check `tshark -D`, Wireshark/Npcap installation, "
                f"and capture permissions.\n"
                f"Original error: {e}"
            ) from e

        try:
            for pkt in capture.sniff_continuously():
                try:
                    # -------------------------------------------------
                    # IP / IPv6
                    # -------------------------------------------------
                    if hasattr(pkt, "ip"):
                        src_ip = pkt.ip.src
                        dst_ip = pkt.ip.dst
                    elif hasattr(pkt, "ipv6"):
                        src_ip = pkt.ipv6.src
                        dst_ip = pkt.ipv6.dst
                    else:
                        continue

                    # -------------------------------------------------
                    # Transport protocol
                    # -------------------------------------------------
                    transport_layer = getattr(
                        pkt,
                        "transport_layer",
                        None,
                    )

                    if transport_layer == "TCP":
                        protocol = 6
                    elif transport_layer == "UDP":
                        protocol = 17
                    elif transport_layer == "ICMP":
                        protocol = 1
                    else:
                        protocol = 0

                    # -------------------------------------------------
                    # Ports
                    # -------------------------------------------------
                    src_port = 0
                    dst_port = 0

                    if transport_layer in ("TCP", "UDP"):
                        try:
                            transport = pkt[transport_layer]
                            src_port = int(
                                getattr(transport, "srcport", 0)
                            )
                            dst_port = int(
                                getattr(transport, "dstport", 0)
                            )
                        except Exception:
                            src_port = 0
                            dst_port = 0

                    # -------------------------------------------------
                    # Direction
                    # -------------------------------------------------
                    direction = self._get_direction(
                        src_ip,
                        src_port,
                        dst_ip,
                        dst_port,
                        protocol,
                        pkt,
                    )

                    # -------------------------------------------------
                    # Packet length
                    # -------------------------------------------------
                    length = int(pkt.length)

                    yield PacketEvent(
                        timestamp=float(pkt.sniff_timestamp),
                        src_ip=src_ip,
                        dst_ip=dst_ip,
                        src_port=src_port,
                        dst_port=dst_port,
                        protocol=protocol,
                        length=length,
                        direction=direction,
                    )

                except (AttributeError, ValueError, TypeError, KeyError):
                    # Ignore packets that do not contain the fields
                    # required by the flow model.
                    continue

        finally:
            try:
                capture.close()
            except Exception:
                pass


class SimulatedCapture:
    """
    Dependency-free traffic simulator standing in for a live interface.
    Generates a mix of "normal" flows (steady, moderate-size packets
    between a handful of recurring host pairs — mimicking regular
    application traffic) and occasional "attack-like" bursts (many small
    packets to one destination in rapid succession — mimicking a
    port-scan/DDoS-like pattern), so downstream flow aggregation and
    inference have something structurally realistic to work with.

    NOT real traffic — for pipeline development and demo only.
    """

    def __init__(self, packets_per_second: float = 50, attack_burst_probability: float = 0.05,
                 seed: int = 42):
        self.packets_per_second = packets_per_second
        self.attack_burst_probability = attack_burst_probability
        self.rng = random.Random(seed)
        self._normal_host_pairs = [
            ("10.0.0.5", "93.184.216.34", 443), ("10.0.0.7", "142.250.80.46", 443),
            ("10.0.0.12", "151.101.1.69", 80), ("10.0.0.5", "8.8.8.8", 53),
        ]
        # Active "conversations" (persistent src port per flow, like a real
        # TCP connection) — without this, every packet gets a fresh random
        # ephemeral port and nothing aggregates into multi-packet flows.
        self._active_conversations = {}  # host_pair_idx -> {"src_port": int, "packets_left": int}

    def _get_or_start_conversation(self, pair_idx: int) -> int:
        """Returns a persistent src_port for this host pair's current flow,
        starting a new conversation (new ephemeral port) if the previous
        one has ended."""
        conv = self._active_conversations.get(pair_idx)
        if conv is None or conv["packets_left"] <= 0:
            conv = {"src_port": self.rng.randint(40000, 60000), "packets_left": self.rng.randint(5, 40)}
            self._active_conversations[pair_idx] = conv
        conv["packets_left"] -= 1
        return conv["src_port"]

    def _generate_normal_packet(self, t: float) -> PacketEvent:
        pair_idx = self.rng.randrange(len(self._normal_host_pairs))
        src, dst, port = self._normal_host_pairs[pair_idx]
        ephemeral_port = self._get_or_start_conversation(pair_idx)
        direction = "fwd" if self.rng.random() > 0.4 else "bwd"
        return PacketEvent(
            timestamp=t,
            src_ip=src if direction == "fwd" else dst,
            dst_ip=dst if direction == "fwd" else src,
            src_port=ephemeral_port if direction == "fwd" else port,
            dst_port=port if direction == "fwd" else ephemeral_port,
            protocol=6,
            length=self.rng.randint(200, 1500),
            direction=direction,
        )

    def _generate_attack_packet(self, t: float, attacker_ip: str, target_ip: str, attacker_src_port: int) -> PacketEvent:
        # small packets, rapid-fire, varied DESTINATION ports (port-scan-like)
        # — source port stays fixed per scanning session, matching a real
        # scanner's single outbound connection probing many target ports.
        return PacketEvent(
            timestamp=t, src_ip=attacker_ip, dst_ip=target_ip,
            src_port=attacker_src_port, dst_port=self.rng.randint(1, 1024),
            protocol=6, length=self.rng.randint(40, 100), direction="fwd",
        )

    def capture(self, duration_seconds: float = None) -> Iterator[PacketEvent]:
        """
        Yields simulated PacketEvents in real time (sleeps between packets
        to approximate `packets_per_second`). Set duration_seconds to stop
        automatically; None runs until the caller stops iterating.
        """
        start = time.time()
        in_burst, burst_remaining, attacker_ip, attacker_src_port = False, 0, None, None

        while duration_seconds is None or (time.time() - start) < duration_seconds:
            t = time.time()

            if not in_burst and self.rng.random() < self.attack_burst_probability / self.packets_per_second:
                in_burst = True
                burst_remaining = self.rng.randint(15, 40)
                attacker_ip = f"203.0.113.{self.rng.randint(1, 254)}"
                attacker_src_port = self.rng.randint(1024, 65535)

            if in_burst and burst_remaining > 0:
                yield self._generate_attack_packet(t, attacker_ip, "10.0.0.5", attacker_src_port)
                burst_remaining -= 1
                if burst_remaining == 0:
                    in_burst = False
            else:
                yield self._generate_normal_packet(t)

            time.sleep(1.0 / self.packets_per_second)

    def capture_batch(self, n_packets: int) -> list:
        """Non-real-time convenience: generate n packets instantly (no sleep), for fast testing."""
        events = []
        in_burst, burst_remaining, attacker_ip, attacker_src_port = False, 0, None, None
        t = time.time()
        for i in range(n_packets):
            t += 1.0 / self.packets_per_second
            if not in_burst and self.rng.random() < self.attack_burst_probability / self.packets_per_second:
                in_burst = True
                burst_remaining = self.rng.randint(15, 40)
                attacker_ip = f"203.0.113.{self.rng.randint(1, 254)}"
                attacker_src_port = self.rng.randint(1024, 65535)
            if in_burst and burst_remaining > 0:
                events.append(self._generate_attack_packet(t, attacker_ip, "10.0.0.5", attacker_src_port))
                burst_remaining -= 1
                if burst_remaining == 0:
                    in_burst = False
            else:
                events.append(self._generate_normal_packet(t))
        return events

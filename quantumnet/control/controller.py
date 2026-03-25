import networkx as nx
from ..utils import Logger

class Controller():
    def __init__(self, network, policy: str = "threshold"):
        self.network = network
        self.hosts = None
        self.policy = policy
        self.control_log = []
        self.logger = Logger.get_instance()

    def _normalize_link(self, alice_id: int, bob_id: int) -> tuple:
        return tuple(sorted((alice_id, bob_id)))

    def _get_link_data(self, alice_id: int, bob_id: int) -> dict:
        edge = self._normalize_link(alice_id, bob_id)
        if not self.network.graph.has_edge(*edge):
            raise KeyError(f'QKD link {edge} not found in topology.')
        return self.network.graph.edges[edge]

    def create_routing_table(self, host_id: int) -> dict:
        """
        Create a routing table for a node in a graph.
        Args:
            host_id (int): The node ID to create the routing table for.
        Returns:
            dict: A routing table for the node.
        """
        shortest_paths = nx.shortest_path(self.network.graph, source=host_id)  # Get shortest paths from the node to all other nodes
        routing_table = {}

        for destination, path in shortest_paths.items():
            if len(path) > 1:  # Ensure there's a valid path
                routing_table[destination] = path  # Store the next hop on the shortest path
            else:
                routing_table[destination] = [host_id]  # Self-routing

        return routing_table

    def register_routing_tables(self):
        """
        Register routing tables for all hosts in the network.
        """
        self.hosts = self.network.hosts

        for host_id in self.hosts:
            routing_table = self.create_routing_table(host_id)
            self.hosts[host_id].set_routing_table(routing_table)

    def check_route(self, route):
        """
        Check if a route is valid.
        Args:
            route (list): A list of nodes in the route.
        Returns:
            bool: True if the route is valid, False otherwise.
        """
        return True

    def announce_to_route_nodes(self, route):
        """
        Announce a message to all nodes in a route.
        Args:
            route (list): A list of nodes in the route.
        """

        if len(route) == 1:
            self.logger.log(f'Node {route[0]} informed.')
        for node in route[1:]:
            self.logger.log(f'Node {node} informed.')

    def announce_to_alice_and_bob(self, route):
        """
        Announce a message to Alice and Bob.
        Args:
            route (list): A list of nodes in the route.
        """

        self.logger.log(f"Alice {route[0]} and Bob {route[-1]} informed.")

    def set_policy(self, policy: str) -> None:
        valid_policies = {"threshold", "on_demand", "hybrid"}
        if policy not in valid_policies:
            raise ValueError(f"Política invalida: {policy}")
        self.policy = policy

    def handle_key_request(self, alice_id: int, bob_id: int, num_bits: int) -> bool:
        if self.policy == "threshold":
            return self._apply_threshold_policy(alice_id, bob_id, num_bits)

        if self.policy == "on_demand":
            return self._apply_on_demand_policy(alice_id, bob_id, num_bits)

        if self.policy == "hybrid":
            return self._apply_hybrid_policy(alice_id, bob_id, num_bits)

        raise ValueError(f"Política não suportada: {self.policy}")
    
    def collect_link_state(self, alice_id: int, bob_id: int) -> dict:
        return self.network.get_qkd_link_state(alice_id, bob_id)
    
    def _log_control_event(self, alice_id: int, bob_id: int, event: str, requested_bits: int = 0, available_before: int = 0, available_after: int = 0, extra: dict | None = None) -> None:
        record = {
            "link": (alice_id, bob_id),
            "policy": self.policy,
            "event": event,
            "requested_bits": requested_bits,
            "available_before": available_before,
            "available_after": available_after,
        }

        if extra:
            record.update(extra)

        self.control_log.append(record)

    def replenish_link(self, alice_id: int, bob_id: int, target_bits: int) -> int:
        """Run BB84 to replenish a link buffer and return the net added bits."""
        before = int(self.collect_link_state(alice_id, bob_id).get('bits_available', 0))
        self.start_bb84_session(alice_id, bob_id, target_bits)
        after = int(self.collect_link_state(alice_id, bob_id).get('bits_available', 0))
        added_bits = max(0, after - before)

        # Count replenishment only when the buffer actually increased.
        if added_bits > 0:
            edge_data = self._get_link_data(alice_id, bob_id)
            edge_data['qkd_replenishment_events'] = int(edge_data.get('qkd_replenishment_events', 0)) + 1
            edge_data['qkd_last_replenishment_amount'] = added_bits

            self._log_control_event(
                alice_id,
                bob_id,
                event="replenish",
                available_before=before,
                available_after=after,
                extra={"target_bits": int(target_bits)},
            )

        return added_bits

    def deny_key_request(self, alice_id: int, bob_id: int, num_bits: int, reason: str) -> bool:
        """Register denial metrics and return False."""
        edge_data = self._get_link_data(alice_id, bob_id)
        edge_data['qkd_denied_requests'] = int(edge_data.get('qkd_denied_requests', 0)) + 1
        edge_data['qkd_policy_last_action'] = f"deny:{reason}"

        state = self.collect_link_state(alice_id, bob_id)
        available = int(state.get('bits_available', 0))

        self._log_control_event(
            alice_id,
            bob_id,
            event="deny",
            requested_bits=int(num_bits),
            available_before=available,
            available_after=available,
            extra={"reason": reason},
        )

        return False

    def serve_key_request(self, alice_id: int, bob_id: int, num_bits: int) -> bool:
        """Consume bits from the link buffer and register control events."""
        before = int(self.collect_link_state(alice_id, bob_id).get('bits_available', 0))
        served = bool(self.network.request_key_from_buffer(alice_id, bob_id, num_bits))
        after = int(self.collect_link_state(alice_id, bob_id).get('bits_available', 0))

        if served:
            self._log_control_event(
                alice_id,
                bob_id,
                event="serve",
                requested_bits=int(num_bits),
                available_before=before,
                available_after=after,
            )

        return served

    def _apply_threshold_policy(self, alice_id: int, bob_id: int, num_bits: int) -> bool:
        state = self.collect_link_state(alice_id, bob_id)
        available = int(state.get('bits_available', 0))
        threshold = int(state.get('min_bits_threshold', 0))

        if available >= num_bits:
            return self.serve_key_request(alice_id, bob_id, num_bits)

        if available < threshold:
            replenish_target = max(threshold, num_bits)
            self.replenish_link(alice_id, bob_id, replenish_target)

        state_after = self.collect_link_state(alice_id, bob_id)
        if int(state_after.get('bits_available', 0)) >= num_bits:
            return self.serve_key_request(alice_id, bob_id, num_bits)

        return self.deny_key_request(alice_id, bob_id, num_bits, "threshold_policy_insufficient_bits")

    def _apply_on_demand_policy(self, alice_id: int, bob_id: int, num_bits: int) -> bool:
        state = self.collect_link_state(alice_id, bob_id)
        available = int(state.get('bits_available', 0))

        if available >= num_bits:
            return self.serve_key_request(alice_id, bob_id, num_bits)

        missing = num_bits - available
        self.replenish_link(alice_id, bob_id, missing)

        state_after = self.collect_link_state(alice_id, bob_id)
        if int(state_after.get('bits_available', 0)) >= num_bits:
            return self.serve_key_request(alice_id, bob_id, num_bits)

        return self.deny_key_request(alice_id, bob_id, num_bits, "on_demand_insufficient_bits")

    def _apply_hybrid_policy(self, alice_id: int, bob_id: int, num_bits: int) -> bool:
        state = self.collect_link_state(alice_id, bob_id)
        available = int(state.get('bits_available', 0))
        threshold = int(state.get('min_bits_threshold', 0))

        if available < threshold:
            self.replenish_link(alice_id, bob_id, threshold)

        state_mid = self.collect_link_state(alice_id, bob_id)
        available_mid = int(state_mid.get('bits_available', 0))

        if available_mid < num_bits:
            missing = num_bits - available_mid
            self.replenish_link(alice_id, bob_id, missing)

        state_after = self.collect_link_state(alice_id, bob_id)
        if int(state_after.get('bits_available', 0)) >= num_bits:
            return self.serve_key_request(alice_id, bob_id, num_bits)

        return self.deny_key_request(alice_id, bob_id, num_bits, "hybrid_insufficient_bits")

    # QKD Discovery Operations
    def get_qkd_nodes(self):
        """List nodes capable of participating in QKD sessions."""
        return sorted(list(self.network.nodes))

    def get_qkd_links(self):
        """List direct links represented as managed QKD resources."""
        links = []
        for edge in self.network.edges:
            data = self.network.graph.edges[edge]
            links.append({
                'link': tuple(sorted(edge)),
                'state': data.get('qkd_state', 'inactive'),
                'supported_protocols': data.get('qkd_supported_protocols', []),
                'bits_available': data.get('qkd_bits_available', 0),
                'key_rate_bps': data.get('qkd_key_rate_bps', 0.0),
            })
        return links

    # QKD Monitoring Operations
    def get_buffer_state(self, alice_id: int, bob_id: int) -> dict:
        """Get current key buffer state for a direct QKD link."""
        data = self._get_link_data(alice_id, bob_id)
        edge = self._normalize_link(alice_id, bob_id)
        return {
            'link': edge,
            'state': data.get('qkd_state', 'inactive'),
            'bits_available': data.get('qkd_bits_available', 0),
            'min_bits_threshold': data.get('qkd_min_bits_threshold', 0),
        }

    def get_link_metrics(self, alice_id: int, bob_id: int) -> dict:
        """Get QKD metrics for a direct link."""
        return self.network.get_qkd_link_state(alice_id, bob_id)

    def get_qkd_sessions(self):
        """Return recorded BB84 session history from application layer."""
        return self.network.application_layer.get_qkd_sessions()

    # QKD Control Operations
    def start_bb84_session(self, alice_id: int, bob_id: int, num_bits: int):
        """Start BB84 and feed generated key bits into the link buffer."""
        result = self.network.application_layer.qkd_bb84_protocol(alice_id, bob_id, num_bits)
        return result

    def request_key(self, alice_id: int, bob_id: int, num_bits: int):
        """Request key material through the configured control policy."""
        served = self.handle_key_request(alice_id, bob_id, num_bits)
        edge = self._normalize_link(alice_id, bob_id)
        action = 'served' if served else 'denied'
        self.logger.log(f'Controller {action} {num_bits} requested bits on QKD link {edge}.')
        return served

    # QKD Management Operations
    def set_minimum_stock(self, alice_id: int, bob_id: int, min_bits: int):
        """Configure the minimum stock threshold for proactive replenishment."""
        data = self._get_link_data(alice_id, bob_id)
        data['qkd_min_bits_threshold'] = max(0, int(min_bits))

    def ensure_minimum_stock(self, default_replenish_bits: int = 256):
        """
        Replenish links that are below threshold using BB84 sessions.

        Returns:
            list[dict]: Operations performed per link.
        """
        actions = []
        for edge in self.network.edges:
            link = tuple(sorted(edge))
            data = self.network.graph.edges[edge]
            threshold = int(data.get('qkd_min_bits_threshold', 0))
            available = int(data.get('qkd_bits_available', 0))

            if available >= threshold:
                continue

            needed = threshold - available
            replenish_bits = max(default_replenish_bits, needed)
            added_bits = self.replenish_link(link[0], link[1], replenish_bits)
            available_after = int(self.collect_link_state(link[0], link[1]).get('bits_available', 0))

            actions.append({
                'link': link,
                'available_before': available,
                'available_after': available_after,
                'threshold': threshold,
                'requested_replenish_bits': replenish_bits,
                'added_bits': added_bits,
                'status': 'replenished' if added_bits > 0 else 'failed',
            })

        return actions

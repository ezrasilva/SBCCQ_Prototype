import networkx as nx
from ..utils import Logger

class Controller():
    def __init__(self, network):
        self.network = network
        self.hosts = None
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
        """Request key material from a link buffer."""
        key_bits = self.network.request_key_from_buffer(alice_id, bob_id, num_bits)
        edge = self._normalize_link(alice_id, bob_id)
        self.logger.log(f'Controller served {len(key_bits)} bits from QKD link {edge}.')
        return key_bits

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
            result = self.start_bb84_session(link[0], link[1], replenish_bits)

            actions.append({
                'link': link,
                'available_before': available,
                'threshold': threshold,
                'requested_replenish_bits': replenish_bits,
                'status': 'started' if result is not None else 'failed',
            })

        return actions

import networkx as nx
from ..utils import Logger
from ..quantum import Qubit
from ..runtime import Clock
from ..config import SimulationConfig
from .host import Host
from ..control.network_context import NetworkContext
from ..control.controller import Controller
from ..layers import *
import random
import os
import csv

class Network():
    """
    An object to use as a network.
    """
    def __init__(self, clock: 'Clock' = None, config: 'SimulationConfig' = None) -> None:
        # Simulation configuration
        self.config = config if config is not None else SimulationConfig()
        # Simulation clock
        self.clock = clock if clock is not None else Clock()
        # Network
        self._graph = nx.Graph()
        self._hosts = {}
        # Execution
        self.logger = Logger.get_instance()
        self.qubit_timeslots = {}  # Dictionary to store created qubits and their timeslots
        # Shared context between layers (without Network reference)
        self._context = NetworkContext(self.clock, self._graph, self._hosts, self.qubit_timeslots, self.config)
        # Layers
        self._physical = PhysicalLayer(self._context)
        self._link = LinkLayer(self._context, self._physical)
        self._network = NetworkLayer(self._context, self._physical)
        self._transport = TransportLayer(self._context, self._network, self._physical)
        self._application = ApplicationLayer(self._context, self._transport)
        self._controller = Controller(self)
        self._application.set_controller(self._controller)
        # Register decoherence as tick callback
        self.clock.on_tick(self._decoherence_on_tick)

    @property
    def hosts(self):
        """
        Dictionary of network hosts. Format: {host_id: host}.

        Returns:
            dict: Dictionary of network hosts.
        """
        return self._hosts

    @property
    def graph(self):
        """
        Network graph.

        Returns:
            nx.Graph: Network graph.
        """
        return self._graph

    @property
    def nodes(self):
        """
        Network graph nodes.

        Returns:
            list: List of graph nodes.
        """
        return self._graph.nodes()

    @property
    def edges(self):
        """
        Network graph edges.

        Returns:
            list: List of graph edges.
        """
        return self._graph.edges()

    # Layers
    @property
    def physical(self):
        """
        Network physical layer.

        Returns:
            PhysicalLayer: Network physical layer.
        """
        return self._physical

    @property
    def linklayer(self):
        """
        Network link layer.

        Returns:
            LinkLayer: Network link layer.
        """
        return self._link

    @property
    def networklayer(self):
        """
        Network layer.

        Returns:
            NetworkLayer: Network layer.
        """
        return self._network

    @property
    def transportlayer(self):
        """
        Transport layer.

        Returns:
            TransportLayer: Transport layer.
        """
        return self._transport

    @property
    def application_layer(self):
        """
        Application layer.

        Returns:
            ApplicationLayer: Application layer.
        """
        return self._application

    @property
    def controller(self):
        """Network controller responsible for QKD resource policies."""
        return self._controller

    def draw(self):
        """
        Draw the network.
        """
        nx.draw(self._graph, with_labels=True)

    def _initialize_channel_metadata(self, edge):
        """Initialize physical and QKD metadata for a link resource."""
        prob_cfg = self.config.probability
        edge_data = self._graph.edges[edge]

        edge_data['prob_on_demand_epr_create'] = random.uniform(prob_cfg.epr_create_min, prob_cfg.epr_create_max)
        edge_data['prob_replay_epr_create'] = random.uniform(prob_cfg.epr_create_min, prob_cfg.epr_create_max)
        edge_data['eprs'] = list()

        # ETSI GS QKD inspired metadata for each link as a managed QKD resource.
        edge_data['qkd_state'] = 'active'
        edge_data['qkd_supported_protocols'] = ['BB84']
        edge_data['qkd_key_buffer'] = list()
        edge_data['qkd_bits_available'] = 0
        # Optional cap for the key buffer (in bits). None means unbounded (legacy behaviour).
        edge_data['qkd_max_buffer_bits'] = None
        edge_data['qkd_key_rate_bps'] = 0.0
        edge_data['qkd_total_generated_bits'] = 0
        edge_data['qkd_total_dropped_bits'] = 0
        edge_data['qkd_total_sessions'] = 0
        edge_data['qkd_successful_sessions'] = 0
        edge_data['qkd_min_bits_threshold'] = 128
        edge_data['qkd_total_consumed_bits'] = 0
        edge_data['qkd_total_requested_bits'] = 0
        edge_data['qkd_served_requests'] = 0
        edge_data['qkd_failed_requests'] = 0
        edge_data['qkd_denied_requests'] = 0
        edge_data['qkd_replenishment_events'] = 0
        edge_data['qkd_policy_last_action'] = None

    def add_host(self, host: Host):
        """
        Add a host to the network hosts dictionary and the host_id to the network graph.

        Args:
            host (Host): The host to be added.
        """
        # Add host to hosts dictionary if it doesn't exist
        if host.host_id not in self._hosts:
            self._hosts[host.host_id] = host
            Logger.get_instance().debug(f'Host {host.host_id} added to network hosts.')
        else:
            raise Exception(f'Host {host.host_id} already exists in network hosts.')

        # Add node to network graph if it doesn't exist
        if not self._graph.has_node(host.host_id):
            self._graph.add_node(host.host_id)
            Logger.get_instance().debug(f'Node {host.host_id} added to network graph.')

        # Add node connections to network graph if they don't exist
        for connection in host.connections:
            if not self._graph.has_edge(host.host_id, connection):
                self._graph.add_edge(host.host_id, connection)
                self._initialize_channel_metadata((host.host_id, connection))
                Logger.get_instance().debug(f'Connections of {host.host_id} added to network graph.')

    def get_host(self, host_id: int) -> Host:
        """
        Return a host from the network.

        Args:
            host_id (int): ID of the host to be returned.

        Returns:
            Host: The host with the given host_id.
        """
        return self._context.get_host(host_id)

    def get_eprs(self):
        """
        Create a list of entangled qubits (EPRs) associated with each graph edge.

        Returns:
            dict: Dictionary where keys are graph edges and values are
              lists of entangled qubits (EPRs) associated with each edge.
        """
        eprs = {}
        for edge in self.edges:
            eprs[edge] = self._graph.edges[edge]['eprs']
        return eprs

    def get_eprs_from_edge(self, alice: int, bob: int) -> list:
        """
        Return the EPRs from a specific edge.

        Args:
            alice (int): Alice host ID.
            bob (int): Bob host ID.
        Returns:
            list: List of EPRs from the edge.
        """
        return self._context.get_eprs_from_edge(alice, bob)

    def remove_epr(self, alice: int, bob: int) -> list:
        """
        Remove an EPR from a channel.

        Args:
            channel (tuple): Communication channel.
        """
        channel = (alice, bob)
        try:
            epr = self._graph.edges[channel]['eprs'].pop(-1)
            return epr
        except IndexError:
            raise Exception('No EPR pairs available.')

    def set_ready_topology(self, topology_name: str, *args: int) -> str:
        """
        Create a graph with one of the ready-to-use topologies.
        Available: Grid, Line, Ring. Nodes are numbered from 0 to n-1, where n is the number of nodes.

        Args:
            topology_name (str): Name of the topology to use.
            **args (int): Arguments for the topology. Usually the number of hosts.

        """
        # Create the graph for the chosen topology
        if topology_name == 'Grade':
            if len(args) != 2:
                raise Exception('Grid topology requires two arguments.')
            self._graph = nx.grid_2d_graph(*args)
        elif topology_name == 'Linha':
            if len(args) != 1:
                raise Exception('Line topology requires one argument.')
            self._graph = nx.path_graph(*args)
        elif topology_name == 'Anel':
            if len(args) != 1:
                raise Exception('Ring topology requires one argument.')
            self._graph = nx.cycle_graph(*args)

        # Convert node labels to integers
        self._graph = nx.convert_node_labels_to_integers(self._graph)
        # Update reference in shared context
        self._context.graph = self._graph

        # Create hosts and add to hosts dictionary
        for node in self._graph.nodes():
            self._hosts[node] = Host(node)
        self.start_hosts()
        self.start_channels()
        self.start_eprs()

    def start_hosts(self, num_qubits: int = None):
        """
        Initialize network hosts.

        Args:
            num_qubits (int): Number of qubits to initialize. If None, uses config.
        """
        if num_qubits is None:
            num_qubits = self.config.defaults.qubits_per_host
        for host_id in self._hosts:
            for i in range(num_qubits):
                self.physical.create_qubit(host_id, increment_qubits=False)
        self.logger.debug("Hosts initialized")

    def start_channels(self):
        """
        Initialize network channels.
        """
        for edge in self.edges:
            self._initialize_channel_metadata(edge)
        self.logger.debug("Channels initialized")

    def get_qkd_link_state(self, alice_id: int, bob_id: int) -> dict:
        """Return QKD resource state for a direct link."""
        edge = tuple(sorted((alice_id, bob_id)))
        if not self._graph.has_edge(*edge):
            raise KeyError(f'QKD link {edge} not found.')

        data = self._graph.edges[edge]
        return {
            'link': edge,
            'state': data.get('qkd_state', 'inactive'),
            'supported_protocols': data.get('qkd_supported_protocols', []),
            'bits_available': data.get('qkd_bits_available', 0),
            'max_buffer_bits': data.get('qkd_max_buffer_bits', None),
            'key_rate_bps': data.get('qkd_key_rate_bps', 0.0),
            'min_bits_threshold': data.get('qkd_min_bits_threshold', 0),
            'total_generated_bits': data.get('qkd_total_generated_bits', 0),
            'total_dropped_bits': data.get('qkd_total_dropped_bits', 0),
            'total_consumed_bits': data.get('qkd_total_consumed_bits', 0),
            'total_requested_bits': data.get('qkd_total_requested_bits', 0),
            'served_requests': data.get('qkd_served_requests', 0),
            'failed_requests': data.get('qkd_failed_requests', 0),
            'denied_requests': data.get('qkd_denied_requests', 0),
            'replenishment_events': data.get('qkd_replenishment_events', 0),
            'policy_last_action': data.get('qkd_policy_last_action', None),
            'total_sessions': data.get('qkd_total_sessions', 0),
            'successful_sessions': data.get('qkd_successful_sessions', 0),
        }

    def _update_consumption_metrics(self, edge_data: dict, num_bits: int, served: bool) -> None:
        """Update QKD accounting after a key consumption request."""
        edge_data['qkd_total_requested_bits'] = int(edge_data.get('qkd_total_requested_bits', 0)) + int(num_bits)

        if served:
            edge_data['qkd_total_consumed_bits'] = int(edge_data.get('qkd_total_consumed_bits', 0)) + int(num_bits)
            edge_data['qkd_served_requests'] = int(edge_data.get('qkd_served_requests', 0)) + 1
            edge_data['qkd_policy_last_action'] = 'serve'
        else:
            edge_data['qkd_failed_requests'] = int(edge_data.get('qkd_failed_requests', 0)) + 1
            edge_data['qkd_policy_last_action'] = 'insufficient_buffer'

    def request_key_from_buffer(self, alice_id: int, bob_id: int, num_bits: int) -> bool:
        """Consume key bits from the link buffer, updating QKD request metrics."""
        if num_bits <= 0:
            return False

        edge = tuple(sorted((alice_id, bob_id)))
        if not self._graph.has_edge(*edge):
            raise KeyError(f'QKD link {edge} not found.')

        edge_data = self._graph.edges[edge]
        buffer_bits = edge_data.get('qkd_key_buffer', [])
        available = len(buffer_bits)

        if available >= num_bits:
            del buffer_bits[:num_bits]
            edge_data['qkd_bits_available'] = len(buffer_bits)
            self._update_consumption_metrics(edge_data, num_bits, served=True)
            return True

        self._update_consumption_metrics(edge_data, num_bits, served=False)
        return False

    def start_eprs(self, num_eprs: int = None):
        """
        Initialize EPR pairs on network edges.

        Args:
            num_eprs (int): Number of EPR pairs to initialize per channel. If None, uses config.
        """
        if num_eprs is None:
            num_eprs = self.config.defaults.eprs_per_channel
        for edge in self.edges:
            for i in range(num_eprs):
                epr = self.physical.create_epr_pair(increment_eprs=False)
                self._graph.edges[edge]['eprs'].append(epr)
                self.logger.debug(f'EPR pair {epr} added to channel.')
        self.logger.debug("EPR pairs added")


    def get_timeslot(self):
        """
        Return the current network timeslot.
        Compatibility wrapper; prefer using self.clock.now directly.

        Returns:
            int: Current network timeslot.
        """
        return self.clock.now

    def register_qubit_creation(self, qubit_id, layer_name):
        """
        Register the creation of a qubit at the current clock timeslot.

        Args:
            qubit_id (int): ID of the created qubit.
            layer_name (str): Name of the layer that created the qubit.
        """
        self._context.register_qubit_creation(qubit_id, layer_name)

    def display_all_qubit_timeslots(self):
        """
        Display the timeslot of all qubits created across different network layers.
        If no qubit was created, displays an appropriate message.
        """
        if not self.qubit_timeslots:
            self.logger.log("No qubits were created.")
        else:
            for qubit_id, info in self.qubit_timeslots.items():
                self.logger.log(f"Qubit {qubit_id} was created at timeslot {info['timeslot']} in layer {info['layer']}")


    def get_total_used_eprs(self):
        """
        Return the total number of EPRs (entangled pairs) used in the network.

        Returns:
            int: Total EPRs used across physical, link and network layers.
        """
        total_eprs = (self._physical.get_used_eprs()+
                      self._link.get_used_eprs() +
                      self._network.get_used_eprs()
        )
        return total_eprs

    def get_total_used_qubits(self):
        """
        Return the total number of qubits used across the entire network.

        Returns:
            int: Total qubits used across physical, link, transport and application layers.
        """

        total_qubits = (self._physical.get_used_qubits() +
                        self._link.get_used_qubits() +
                        self._transport.get_used_qubits() +
                        self._application.get_used_qubits()

        )
        return total_qubits
    
    def get_metrics(self, metrics_requested=None, output_type="csv", file_name="metrics_output.csv"):
            """
            Retrieve network metrics as requested and export, print, or store them.

            Args:
                metrics_requested: List of metrics to return (optional).
                                If None, all metrics will be included.
                output_type: Specifies how metrics should be returned.
                            "csv" to export as CSV file (default),
                            "print" to display on console,
                            "variable" to return metrics as a variable.
                file_name: CSV file name (used only when output_type="csv").

            Returns:
                If output_type is "variable", returns a dictionary with the requested metrics.
            """
            # Dictionary with all available metrics
            available_metrics = {
                "Total Timeslot": self.get_timeslot(),
                "Used EPRs": self.get_total_used_eprs(),
                "Used Qubits": self.get_total_used_qubits(),
                "Transport Layer Fidelity": self.transportlayer.avg_fidelity_on_transportlayer(),
                "Link Layer Fidelity": self.linklayer.avg_fidelity_on_linklayer(),
                "Average Routes": self.networklayer.get_avg_size_routes()
            }

            # If no specific metrics were requested, use all
            if metrics_requested is None:
                metrics_requested = available_metrics.keys()

            # Filter requested metrics
            metrics = {metric: available_metrics[metric] for metric in metrics_requested if metric in available_metrics}

            # Handle output based on requested type
            if output_type == "print":
                for metric, value in metrics.items():
                    self.logger.log(f"{metric}: {value}")
            elif output_type == "csv":
                current_directory = os.getcwd()
                file_path = os.path.join(current_directory, file_name)
                with open(file_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Metric', 'Value'])
                    for metric, value in metrics.items():
                        writer.writerow([metric, value])
                self.logger.log(f"Metrics successfully exported to {file_path}")
            elif output_type == "variable":
                return metrics
            else:
                raise ValueError("Invalid output type. Choose between 'print', 'csv', or 'variable'.")

    def _decoherence_on_tick(self, clock):
        """
        Tick callback: apply decoherence to all qubits and EPRs.
        Automatically called by the clock on each tick().
        """
        decoherence_factor = self.config.decoherence.per_timeslot
        current_timeslot = clock.now

        # Apply decoherence to qubits in each host
        for host_id, host in self.hosts.items():
            for qubit in host.memory:
                if qubit.qubit_id in self.qubit_timeslots:
                    creation_timeslot = self.qubit_timeslots[qubit.qubit_id]['timeslot']
                    if creation_timeslot < current_timeslot:
                        current_fidelity = qubit.get_current_fidelity()
                        new_fidelity = current_fidelity * decoherence_factor
                        qubit.set_current_fidelity(new_fidelity)

        # Apply decoherence to EPRs in all channels (network edges)
        for edge in self.edges:
            if 'eprs' in self._graph.edges[edge]:
                for epr in self._graph.edges[edge]['eprs']:
                    current_fidelity = epr.get_current_fidelity()
                    new_fidelity = current_fidelity * decoherence_factor
                    epr.set_fidelity(new_fidelity)

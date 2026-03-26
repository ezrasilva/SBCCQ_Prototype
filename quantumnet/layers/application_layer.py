import random
from quantumnet.topology import Host
from quantumnet.quantum import Qubit
from quantumnet.utils import Logger

class ApplicationLayer:
    def __init__(self, context, transport_layer):
        """
        Initialize the QKD (Quantum Key Distribution) application layer.

        Args:
            context (NetworkContext): Shared network context.
            transport_layer (TransportLayer): Network transport layer.
        """
        self._context = context
        self._transport_layer = transport_layer
        self.logger = Logger.get_instance()
        self.used_qubits = 0
        self._qkd_sessions = []
        self._next_session_id = 1
        self._controller = None

    def __str__(self):
        return 'Application Layer'

    def get_used_qubits(self):
        """
        Return the number of qubits used in the application layer and log the information.

        Returns:
            int: Number of qubits used in the application layer.
        """
        self.logger.debug(f"Qubits used in layer {self.__class__.__name__}: {self.used_qubits}")
        return self.used_qubits

    def get_qkd_sessions(self):
        """Return all BB84 session records created by this layer."""
        return list(self._qkd_sessions)

    def _record_qkd_session(self, session_data):
        """Store a QKD session entry and advance internal session id."""
        record = dict(session_data)
        record['session_id'] = self._next_session_id
        self._qkd_sessions.append(record)
        self._next_session_id += 1
        return record

    def _get_qkd_link_data(self, alice_id, bob_id):
        """Return graph edge metadata for a direct QKD link if available."""
        edge = tuple(sorted((alice_id, bob_id)))
        if not self._context.graph.has_edge(*edge):
            return None
        return self._context.graph.edges[edge]

    def run_app(self, app_name, *args):
        """
        Execute the desired application by the given name.

        Args:
            app_name (str): The name of the application to execute.
            *args: Variable arguments for the specific application: alice_id, bob_id, and num_qubits.
        """
        if app_name == "QKD_E91":
            alice_id, bob_id, num_qubits = args
            return self.qkd_e91_protocol(alice_id,bob_id, num_qubits)
        if app_name == "QKD_BB84" or app_name == "BB84":
            alice_id, bob_id, num_bits = args
            return self.qkd_bb84_protocol(alice_id, bob_id, num_bits)
        if app_name == "QKD_REQUEST_KEY":
            alice_id, bob_id, num_bits = args
            return self.request_qkd_key(alice_id, bob_id, num_bits)
        else:
            self.logger.log(f"Application not executed or not found.")
            return False

    def set_controller(self, controller):
        """Attach a controller used to manage QKD key requests."""
        self._controller = controller

    def request_qkd_key(self, alice_id, bob_id, num_bits):
        """Request key bits via control policy (Application -> Controller -> Network)."""
        if self._controller is None:
            raise RuntimeError('Controller not configured for ApplicationLayer.')
        return self._controller.handle_key_request(alice_id, bob_id, num_bits)


    def prepare_e91_qubits(self, key, bases):
        """
        Prepare qubits according to the key and bases provided for the E91 protocol.

        Args:
            key (list): Key containing the bit sequence.
            bases (list): Bases used to measure the qubits.

        Returns:
            list: List of prepared qubits.
        """
        self._context.clock.emit('e91_qubits_prepared', num_qubits=len(key))
        self.logger.debug(f"E91 qubits prepared at timeslot: {self._context.clock.now}")
        qubits = []
        for bit, base in zip(key, bases):
            qubit = Qubit(qubit_id=random.randint(0, 1000))  # Create new qubit with random ID
            if bit == 1:
                qubit.apply_x()  # Apply X gate (NOT) to qubit if bit is 1
            if base == 1:
                qubit.apply_hadamard()  # Apply Hadamard gate to qubit if base is 1
            qubits.append(qubit)  # Add prepared qubit to list
        return qubits

    def apply_bases_and_measure_e91(self, qubits, bases):
        """
        Apply measurement bases and measure qubits in the E91 protocol.

        Args:
            qubits (list): List of qubits to be measured.
            bases (list): List of bases to apply for measurement.

        Returns:
            list: Measurement results.
        """
        self._context.clock.emit('e91_measurement', num_qubits=len(qubits))
        self.logger.debug(f"E91 measurements performed at timeslot: {self._context.clock.now}")
        results = []
        for qubit, base in zip(qubits, bases):
            if base == 1:
                qubit.apply_hadamard()  # Apply Hadamard gate before measurement if base is 1
            measurement = qubit.measure()  # Measure the qubit
            results.append(measurement)  # Add measurement result to results list
        return results

    def qkd_e91_protocol(self, alice_id, bob_id, num_bits):
        """
        Implement the E91 protocol for Quantum Key Distribution (QKD).

        Args:
            alice_id (int): Alice host ID.
            bob_id (int): Bob host ID.
            num_bits (int): Number of bits for the key.

        Returns:
            list: Final key generated by the protocol, or None if transmission fails.
        """
        alice = self._context.get_host(alice_id)  # Get Alice's host
        bob = self._context.get_host(bob_id)  # Get Bob's host

        final_key = []  # Initialize final key

        while len(final_key) < num_bits:
            num_qubits = int((num_bits - len(final_key)) * 2)  # Calculate number of qubits needed
            self.used_qubits += num_qubits
            self.logger.log(f'Starting E91 protocol with {num_qubits} qubits.')

            # Step 1: Alice prepares the qubits
            key = [random.choice([0, 1]) for _ in range(num_qubits)]  # Generate random bit key
            bases_alice = [random.choice([0, 1]) for _ in range(num_qubits)]  # Generate random measurement bases for Alice
            qubits = self.prepare_e91_qubits(key, bases_alice)  # Prepare qubits based on key and bases
            self.logger.log(f'Qubits prepared with key: {key} and bases: {bases_alice}')

            # Step 2: Transmit qubits from Alice to Bob
            success = self._transport_layer.run_transport_layer(alice_id, bob_id, num_qubits)
            if not success:
                self.logger.log(f'Failed to transmit qubits from Alice to Bob.')
                return None

            self._context.clock.tick()  # E91 protocol round costs time
            self.logger.debug(f"E91 round completed at timeslot: {self._context.clock.now}")

            # Step 3: Bob chooses random bases and measures the qubits
            bases_bob = [random.choice([0, 1]) for _ in range(num_qubits)]  # Generate random measurement bases for Bob
            results_bob = self.apply_bases_and_measure_e91(qubits, bases_bob)  # Bob measures qubits using his bases
            self.logger.log(f'Measurement results: {results_bob} with bases: {bases_bob}')

            # Step 4: Alice and Bob share their bases and find common indices
            common_indices = [i for i in range(len(bases_alice)) if bases_alice[i] == bases_bob[i]]  # Indices where bases match
            self.logger.log(f'Common indices: {common_indices}')

            # Step 5: Key extraction based on common indices
            shared_key_alice = [key[i] for i in common_indices]  # Shared key generated by Alice
            shared_key_bob = [results_bob[i] for i in common_indices]  # Shared key generated by Bob

            # Step 6: Verify if keys match
            for a, b in zip(shared_key_alice, shared_key_bob):
                if a == b and len(final_key) < num_bits:  # Limit final key size
                    final_key.append(a)

            self.logger.log(f"Keys obtained so far: {final_key}")

            if len(final_key) >= num_bits:
                final_key = final_key[:num_bits]  # Ensure final key has the exact requested size
                self.logger.log(f"E91 protocol succeeded. Final shared key: {final_key}")
                return final_key

        return None

    def prepare_bb84_qubits(self, key_bits, bases):
        """
        Prepare BB84 qubits based on random key bits and Alice's bases.

        Encoding:
        - Base 0 (Z): bit 0 -> |0>, bit 1 -> |1>
        - Base 1 (X): bit 0 -> |+>, bit 1 -> |->

        Args:
            key_bits (list[int]): Random bits generated by Alice.
            bases (list[int]): Alice bases (0=Z, 1=X).

        Returns:
            list[Qubit]: Prepared qubits.
        """
        qubits = []
        for bit, base in zip(key_bits, bases):
            qubit = Qubit(qubit_id=random.randint(0, 1000000))

            if base == 0:
                if bit == 1:
                    qubit.apply_x()
            else:
                if bit == 1:
                    qubit.apply_x()
                qubit.apply_hadamard()

            qubits.append(qubit)

        self._context.clock.emit('bb84_qubits_prepared', num_qubits=len(qubits))
        return qubits

    def measure_bb84_qubits(self, qubits, bases):
        """
        Measure BB84 qubits according to Bob's chosen bases.

        Args:
            qubits (list[Qubit]): Received qubits.
            bases (list[int]): Bob bases (0=Z, 1=X).

        Returns:
            list[int]: Bob's measurement outcomes.
        """
        results = []
        for qubit, base in zip(qubits, bases):
            if base == 1:
                qubit.apply_hadamard()
            results.append(qubit.measure())

        self._context.clock.emit('bb84_measurement', num_qubits=len(results))
        return results

    def decode_bb84_results(self, key_bits, bases_alice, bases_bob, received_qubits):
        """
        Decode Bob's BB84 measurements using basis matching and channel fidelity.

        The current Qubit abstraction uses stochastic Hadamard behavior, so direct
        gate-level simulation can inject artificial noise. This decoder keeps the
        BB84 logic at protocol level while still reflecting transport quality.

        Args:
            key_bits (list[int]): Alice encoded bits.
            bases_alice (list[int]): Alice chosen bases (0=Z, 1=X).
            bases_bob (list[int]): Bob chosen bases (0=Z, 1=X).
            received_qubits (list[Qubit]): Qubits received by Bob after transport.

        Returns:
            list[int]: Bob measured bits.
        """
        results = []
        for bit, base_a, base_b, qubit in zip(key_bits, bases_alice, bases_bob, received_qubits):
            if base_a != base_b:
                # In BB84, different bases produce random outcomes.
                results.append(random.choice([0, 1]))
                continue

            fidelity = qubit.get_current_fidelity()
            fidelity = max(0.0, min(1.0, fidelity))

            # Convert transport fidelity into a moderate bit-flip probability.
            # Higher fidelity means fewer errors; capped to avoid pathological rates.
            bit_flip_probability = min(0.25, (1.0 - fidelity) * 0.35)
            if random.random() < bit_flip_probability:
                results.append(1 - bit)
            else:
                results.append(bit)

        self._context.clock.emit('bb84_measurement', num_qubits=len(results))
        return results

    def qkd_bb84_protocol(self, alice_id, bob_id, num_bits, qber_threshold=0.15, sample_ratio=0.2, max_rounds=160, max_failed_rounds=48):
        """
        Implement BB84 protocol for Quantum Key Distribution (QKD).

        Args:
            alice_id (int): Alice host ID.
            bob_id (int): Bob host ID.
            num_bits (int): Desired final shared key size.
            qber_threshold (float): Abort threshold for estimated QBER.
            sample_ratio (float): Fraction of sifted bits used for QBER estimation.
            max_rounds (int): Maximum protocol rounds.
            max_failed_rounds (int): Maximum failed rounds (transport/high-QBER/invalid).

        Returns:
            dict|None: Dictionary with final key and protocol data, or None on failure.
        """
        alice = self._context.get_host(alice_id)
        bob = self._context.get_host(bob_id)
        session_start = self._context.clock.now

        final_key = []
        rounds = 0
        qber_history = []
        failed_rounds = 0

        if max_failed_rounds is None:
            max_failed_rounds = max_rounds

        while len(final_key) < num_bits and rounds < max_rounds:
            rounds += 1
            remaining_bits = num_bits - len(final_key)
            num_qubits = max(4, remaining_bits * 2)
            self.used_qubits += num_qubits

            key_bits = [random.choice([0, 1]) for _ in range(num_qubits)]
            bases_alice = [random.choice([0, 1]) for _ in range(num_qubits)]
            qubits = self.prepare_bb84_qubits(key_bits, bases_alice)

            # run_transport_layer requires Alice to have exactly num_qubits in memory.
            alice.memory.clear()
            for qubit in qubits:
                alice.add_qubit(qubit)

            bob_initial_memory_size = len(bob.memory)
            success = self._transport_layer.run_transport_layer(alice_id, bob_id, num_qubits)
            if not success:
                self.logger.log('BB84 transport failed in this round; retrying in next round.')
                self._context.clock.emit('bb84_round_failed', reason='transport_failed', round=rounds)
                failed_rounds += 1
                self._context.clock.tick()
                if failed_rounds >= max_failed_rounds:
                    break
                continue

            received_qubits = bob.memory[bob_initial_memory_size:]
            if len(received_qubits) < num_qubits:
                self.logger.log('BB84 round invalid: Bob received fewer qubits than expected; retrying.')
                self._context.clock.emit('bb84_round_failed', reason='insufficient_received_qubits', round=rounds)
                failed_rounds += 1
                self._context.clock.tick()
                if failed_rounds >= max_failed_rounds:
                    break
                continue

            bases_bob = [random.choice([0, 1]) for _ in range(num_qubits)]
            results_bob = self.decode_bb84_results(key_bits, bases_alice, bases_bob, received_qubits)

            sifted_indices = [i for i in range(num_qubits) if bases_alice[i] == bases_bob[i]]
            sifted_alice = [key_bits[i] for i in sifted_indices]
            sifted_bob = [results_bob[i] for i in sifted_indices]

            if not sifted_alice:
                self.logger.log('BB84 round produced no sifted bits; retrying next round.')
                failed_rounds += 1
                self._context.clock.tick()
                if failed_rounds >= max_failed_rounds:
                    break
                continue

            if len(sifted_alice) == 1:
                sample_size = 1
            else:
                sample_size = max(1, int(len(sifted_alice) * sample_ratio))
                sample_size = min(sample_size, len(sifted_alice) - 1)

            sample_positions = set(random.sample(range(len(sifted_alice)), sample_size))
            sample_errors = sum(1 for i in sample_positions if sifted_alice[i] != sifted_bob[i])
            qber_estimate = sample_errors / sample_size if sample_size > 0 else 0.0
            qber_history.append(qber_estimate)

            if qber_estimate > qber_threshold:
                self.logger.log(f'BB84 high QBER in round {rounds}: {qber_estimate}; discarding this round and retrying.')
                self._context.clock.emit('bb84_round_failed', reason='high_qber', qber=qber_estimate, round=rounds)
                failed_rounds += 1
                self._context.clock.tick()
                if failed_rounds >= max_failed_rounds:
                    break
                continue

            key_positions = [i for i in range(len(sifted_alice)) if i not in sample_positions]
            if not key_positions:
                failed_rounds += 1
                self._context.clock.emit('bb84_round_failed', reason='no_key_positions', round=rounds)
                self._context.clock.tick()
                if failed_rounds >= max_failed_rounds:
                    break
                continue

            # Keep non-sampled bits as candidate key material.
            # In practical BB84, remaining errors are corrected later by reconciliation.
            residual_errors = sum(1 for i in key_positions if sifted_alice[i] != sifted_bob[i])
            residual_qber = residual_errors / len(key_positions)
            if residual_qber > qber_threshold:
                self.logger.log(
                    f'BB84 residual QBER too high in round {rounds}: {residual_qber}; discarding round.'
                )
                self._context.clock.emit(
                    'bb84_round_failed',
                    reason='residual_high_qber',
                    qber=residual_qber,
                    round=rounds,
                )
                failed_rounds += 1
                self._context.clock.tick()
                if failed_rounds >= max_failed_rounds:
                    break
                continue

            candidate_bits = [sifted_alice[i] for i in key_positions]

            for bit in candidate_bits:
                if len(final_key) < num_bits:
                    final_key.append(bit)

            self._context.clock.emit(
                'bb84_round_complete',
                round=rounds,
                sifted=len(sifted_alice),
                sample_size=sample_size,
                qber=qber_estimate,
                accumulated_key=len(final_key),
                failed_rounds=failed_rounds,
            )
            self._context.clock.tick()

        if len(final_key) < num_bits:
            abort_reason = 'max_rounds_reached' if rounds >= max_rounds else 'max_failed_rounds_reached'
            self.logger.log(f'BB84 failed: {abort_reason} before obtaining final key size.')
            self._context.clock.emit('bb84_aborted', reason=abort_reason, round=rounds, failed_rounds=failed_rounds)
            link_data = self._get_qkd_link_data(alice_id, bob_id)
            if link_data is not None:
                link_data['qkd_total_sessions'] += 1
                link_data['qkd_state'] = 'inactive'
            session_record = self._record_qkd_session({
                'protocol': 'BB84',
                'participants': (alice_id, bob_id),
                'started_at': session_start,
                'ended_at': self._context.clock.now,
                'generated_bits': 0,
                'requested_bits': num_bits,
                'status': 'failed',
                'rounds': rounds,
                'failed_rounds': failed_rounds,
                'abort_reason': abort_reason,
            })
            return None

        final_key = final_key[:num_bits]
        avg_qber = sum(qber_history) / len(qber_history) if qber_history else 0.0

        self.logger.log(f'BB84 succeeded. Final shared key size={len(final_key)} avg_qber={avg_qber}')
        self._context.clock.emit('bb84_complete', alice=alice_id, bob=bob_id, key_size=len(final_key), avg_qber=avg_qber)

        # Persist key material in link buffer to turn BB84 into a key producer.
        link_data = self._get_qkd_link_data(alice_id, bob_id)
        buffered_bits = 0
        if link_data is not None:
            # Total generated bits counts what BB84 produced, even if the buffer is capped.
            link_data['qkd_total_generated_bits'] += len(final_key)

            buffer_bits = link_data.get('qkd_key_buffer', [])
            max_buffer_bits = link_data.get('qkd_max_buffer_bits', None)

            dropped_bits = 0
            if max_buffer_bits is None:
                buffer_bits.extend(final_key)
            else:
                cap = max(0, int(max_buffer_bits))
                free = max(0, cap - len(buffer_bits))
                if free <= 0:
                    dropped_bits = len(final_key)
                else:
                    buffer_bits.extend(final_key[:free])
                    dropped_bits = max(0, len(final_key) - free)

            if dropped_bits:
                link_data['qkd_total_dropped_bits'] = int(link_data.get('qkd_total_dropped_bits', 0)) + int(dropped_bits)

            link_data['qkd_key_buffer'] = buffer_bits
            buffered_bits = len(buffer_bits)
            link_data['qkd_bits_available'] = buffered_bits
            link_data['qkd_total_sessions'] += 1
            link_data['qkd_successful_sessions'] += 1
            elapsed_slots = max(1, self._context.clock.now - session_start)
            link_data['qkd_key_rate_bps'] = len(final_key) / elapsed_slots
            link_data['qkd_state'] = 'active'
        else:
            self.logger.log(f'BB84 key generated for ({alice_id}, {bob_id}), but no direct QKD edge exists to buffer it.')

        session_record = self._record_qkd_session({
            'protocol': 'BB84',
            'participants': (alice_id, bob_id),
            'started_at': session_start,
            'ended_at': self._context.clock.now,
            'generated_bits': len(final_key),
            'requested_bits': num_bits,
            'status': 'completed',
            'rounds': rounds,
            'failed_rounds': failed_rounds,
            'avg_qber': avg_qber,
        })

        return {
            'key': final_key,
            'qber_history': qber_history,
            'avg_qber': avg_qber,
            'rounds': rounds,
            'failed_rounds': failed_rounds,
            'session_id': session_record['session_id'],
            'buffered_bits': buffered_bits,
        }

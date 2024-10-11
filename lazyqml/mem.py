import psutil
import pennylane as qml

def available_memory():
    """
    Check the available system memory in bytes using psutil.
    Returns:
        available_mem (int): Available RAM in bytes.
    """
    mem_info = psutil.virtual_memory()
    return mem_info.available

def estimate_required_memory(num_qubits, overhead_factor=1.5):
    """
    Estimate the required memory for a given number of qubits in a PennyLane circuit.
    Args:
        num_qubits (int): Number of qubits in the circuit.
        overhead_factor (float): Factor to account for memory overhead.
    Returns:
        required_memory (int): Estimated required memory in bytes.
    """
    # Size of the state vector in bytes: 16 bytes per complex number
    raw_memory = 16 * (2 ** num_qubits)
    # Add an overhead to account for non-optimized memory usage
    required_memory = int(overhead_factor * raw_memory)
    return required_memory

def check_and_run_circuit(circuit, num_qubits):
    """
    Check if the system has sufficient memory to run a PennyLane circuit.
    Args:
        circuit (qml.QNode): The PennyLane circuit to be executed.
        num_qubits (int): Number of qubits in the circuit.
    Returns:
        result (any): Result of the circuit execution if memory is sufficient.
    """
    # Estimate the required memory
    required_mem = estimate_required_memory(num_qubits)
    # Check the available memory
    available_mem = available_memory()

    # Print memory details
    print(f"Available Memory: {available_mem / (1024 ** 3):.2f} GB")
    print(f"Estimated Required Memory: {required_mem / (1024 ** 3):.2f} GB")

    # Check if there is sufficient memory
    if required_mem > available_mem:
        print("Insufficient memory to run the circuit. Aborting execution.")
        return False
    else:
        print("Sufficient memory available. Executing circuit...")
        return True


def calculate_quantum_memory(num_qubits, overhead=2):
    # Each qubit state requires 2 complex numbers (amplitude and phase)
    # Each complex number uses 2 double-precision floats (16 bytes)
    bytes_per_qubit_state = 16
    
    # Number of possible states is 2^n, where n is the number of qubits
    num_states = 2 ** num_qubits
    
    # Total memory in bytes
    total_memory_bytes = num_states * bytes_per_qubit_state * overhead
    
    # Convert to more readable units

    return total_memory_bytes / (1024**3)

    
    

def print_memory_requirements(num_qubits):
    memory = calculate_quantum_memory(num_qubits)
    print(f"Memory requirements for {num_qubits} qubits:")
    print(f"Bytes: {memory['bytes']:.2f}")
    print(f"Kilobytes: {memory['kilobytes']:.2f}")
    print(f"Megabytes: {memory['megabytes']:.2f}")
    print(f"Gigabytes: {memory['gigabytes']:.2f}")
    print(f"Terabytes: {memory['terabytes']:.2f}")

# Example usage
for qubits in [4, 8, 16, 24, 32, 48]:
    print_memory_requirements(qubits)
    print()
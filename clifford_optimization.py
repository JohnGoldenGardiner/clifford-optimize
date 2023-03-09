import numpy as np
from qiskit import QuantumCircuit
from qiskit.algorithms import VQEResult
from qiskit.opflow import PauliSumOp


class PauliSum:
    """
    Objects in this class describe a sum of Pauli strings
    """
    def __init__(self):
        self.S_z = None
        self.S_x = None
        self.coefficients = None

    def get_PauliSumOp(self) -> PauliSumOp:
        """
        Translate to a Qiskit qiskit.opflow.PauliSumOp object
        """
        
        pauli_names = ['I', 'Z', 'X', 'Y']

        num_terms = self.S_z.shape[1]
        num_qubits = self.S_z.shape[0]

        pauli_list = []
        for term in range(num_terms):
            pauli_string = ''
            for qubit in range(num_qubits):
                i = 2*self.S_x[qubit, term] + self.S_z[qubit, term]
                pauli_string += pauli_names[i]
                # Reverse this to match Qiskit ordering (qubit 0 is rightmost)
            
            pauli_string = pauli_string[::-1]

            coefficient = self.coefficients[term]

            pauli_list.append((pauli_string, coefficient))

        return PauliSumOp.from_list(pauli_list)

    def to_matrix(self, qiskit_ordering=True) -> np.ndarray:
        """
        Output the matrix form of the sum of Pauli strings

        Args:
            qiskit_ordering: use the qiskit ordering where the 0th qubit
                is rightmost (i.e. innermost) in the Kronecker product
        """
        
        num_terms = self.S_z.shape[1]
        num_qubits = self.S_z.shape[0]
        
        paulis = np.array(
            [[[1, 0],
              [0, 1]],
             [[1, 0],
              [0, -1]],
             [[0, 1],
              [1, 0]],
             [[0, -1j],
              [1j, 0]]]
        )

        mat = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
        for term in range(num_terms):

            print(f'Starting term {term + 1} out of {num_terms}', end='\r')

            mat_term = np.eye(1)
            for qubit in range(num_qubits):
                i = 2* self.S_x[qubit, term] + self.S_z[qubit, term]
                if qiskit_ordering:
                    mat_term = np.kron(paulis[i], mat_term)
                else:
                    mat_term = np.kron(mat_term, paulis[i])

            mat += self.coefficients[term]*mat_term


        return mat


def parse_circuit(circuit: QuantumCircuit):
    """
    Get gates and qubit count from a qiskit QuantumCircuit object

    Returns:
        n: number of qubits

        gate_list: list of gates where each gate is a triple with the gate
            name (string), the qubits acted on (tuple) and the index of 
            the gate parameter (int), if it is parametrized
    """

    n = circuit.num_qubits
    qubits = circuit.qubits

    gate_list = []
    for instruction in circuit.data:

        name = instruction.operation.name
        q = tuple(qubits.index(qubit) for qubit in instruction.qubits)
        
        param_list = instruction.operation.params
        if param_list != []:
            param_index = param_list[0].index
        else:
            param_index = None
        
        gate = (name, q, param_index)

        gate_list.append(gate)

    return n, gate_list


def cliffordize(gate_list, discrete_params):
    """
    Replaces parametrized single-qubit rotation gates with Clifford gates.

    Args:
        discrete_params: array of integers form 0 to 3 representing rotation
            angles 0, pi/2, pi, 3*pi/2 for rotation gates.
    """

    z_gates = ['id', 's', 'z', 'sdg']
    x_gates = ['id', 'sx', 'x', 'sxdg']
    y_gates = ['id', 'sy', 'y', 'sydg']
    
    clifford_gate_list = []
    for gate in gate_list:

        if gate[0] == 'rz':
            name = z_gates[discrete_params[gate[2]]]

        elif gate[0] == 'rx':
            name = x_gates[discrete_params[gate[2]]]

        elif gate[0] == 'ry':
            name = y_gates[discrete_params[gate[2]]]
        
        else:
            name = gate[0]

        q = gate[1]

        clifford_gate_list.append((name, q))

    return clifford_gate_list


def stabilizer_evolve(init_pauli_sum: PauliSum, circuit, reverse=False):
    """
    Conjugates a sum of Pauli strings by a list of Clifford gates

    Args:
        init_pauli_sum: PauliSum object to be conjugated

        circuit: Clifford gates to conjugate init_pauli_sum by. Should be 
            list of Clifford gates as output by cliffordize().

        reverse: if True, conjugate init_pauli_sum by the Hermitian conjugate 
            of circuit
    """

    init_S_z = init_pauli_sum.S_z
    init_S_x = init_pauli_sum.S_x
    init_signs = np.zeros(init_S_z.shape[1], dtype=int)

    S_z = init_S_z.copy()
    S_x = init_S_x.copy()
    signs = init_signs.copy()

    if reverse == True:
        daggers = {
            'id' : 'id', 's' : 'sdg', 'sdg' : 's', 'sx' : 'sxdg',
            'sxdg' : 'sx', 'sy' : 'sydg', 'sydg' : 'sy', 'h' : 'h',
            'z' : 'z', 'x' : 'x', 'y' : 'y', 'cz' : 'cz', 'cx' : 'cx',
            'cy' : 'cy', 'swap' : 'swap'
        }
        new_circuit = []
        for gate in circuit[::-1]:
            new_circuit.append((daggers[gate[0]], gate[1]))
    else:
        new_circuit = circuit.copy()

    # Evolve
    for gate in new_circuit:
        
        q = gate[1]

        # Single-qubit gates
        if gate[0] == 'id':
            pass

        elif gate[0] == 's':
            signs = (signs + S_z[q[0]]*S_x[q[0]]) % 2
            S_z[q[0]] = (S_z[q[0]] + S_x[q[0]]) % 2
            

        elif gate[0] == 'sdg':
            signs = (signs + S_x[q[0]] + S_z[q[0]]*S_x[q[0]]) % 2
            S_z[q[0]] = (S_z[q[0]] + S_x[q[0]]) % 2
            

        elif gate[0] == 'sx':
            signs = (signs + S_z[q[0]] + S_z[q[0]]*S_x[q[0]]) % 2
            S_x[q[0]] = (S_x[q[0]] + S_z[q[0]]) % 2
            

        elif gate[0] == 'sxdg':
            signs = (signs + S_z[q[0]]*S_x[q[0]]) % 2
            S_x[q[0]] = (S_x[q[0]] + S_z[q[0]]) % 2
            

        elif gate[0] == 'sy':
            signs = (signs + S_x[q[0]] + S_z[q[0]]*S_x[q[0]]) % 2
            S_z[q[0]], S_x[q[0]] = S_x[q[0]].copy(), S_z[q[0]].copy()
            

        elif gate[0] == 'sydg':
            signs = (signs + S_z[q[0]] + S_z[q[0]]*S_x[q[0]]) % 2
            S_z[q[0]], S_x[q[0]] = S_x[q[0]].copy(), S_z[q[0]].copy()
            

        elif gate[0] == 'h':
            signs = (signs + S_z[q[0]]*S_x[q[0]]) % 2
            S_z[q[0]], S_x[q[0]] = S_x[q[0]].copy(), S_z[q[0]].copy()
            

        elif gate[0] == 'z':
            signs = (signs + S_x[q[0]]) % 2

        elif gate[0] == 'x':
            signs = (signs + S_z[q[0]]) % 2

        elif gate[0] == 'y':
            signs = (signs + S_z[q[0]] + S_x[q[0]]) % 2

        # Two-qubit gates
        elif gate[0] == 'cz':
            signs = (signs + S_x[q[0]]*S_x[q[1]]*(S_z[q[0]]^S_z[q[1]])) % 2
            S_z[q[0]] = (S_z[q[0]] + S_x[q[1]]) % 2
            S_z[q[1]] = (S_z[q[1]] + S_x[q[0]]) % 2

        elif gate[0] == 'cx':
            signs = (signs + S_x[q[0]]*S_z[q[1]]*(S_z[q[0]] == S_x[q[1]])) % 2
            S_z[q[0]] = (S_z[q[0]] + S_z[q[1]]) % 2
            S_x[q[1]] = (S_x[q[1]] + S_x[q[0]]) % 2

        elif gate[0] == 'cy':
            signs = (signs + S_x[q[0]]*(S_z[q[1]]^S_x[q[1]])*(S_z[q[0]] == S_z[q[1]])) % 2
            S_z[q[0]] = (S_z[q[0]] + S_z[q[1]] + S_x[q[1]]) % 2
            S_z[q[1]] = (S_z[q[1]] + S_x[q[0]]) % 2
            S_x[q[1]] = (S_x[q[1]] + S_x[q[0]]) % 2

        elif gate[0] == 'swap':
            S_z[q[0]], S_z[q[1]] = S_z[q[1]], S_z[q[0]]
            S_x[q[0]], S_x[q[1]] = S_x[q[1]], S_x[q[0]]

        # Catch all other gates
        else:
            print(f'Warning: Encountered unimplemented gate: {gate[0]}')

    pauli_sum = PauliSum()
    pauli_sum.S_z = S_z
    pauli_sum.S_x = S_x
    pauli_sum.coefficients = ((-1)**signs) * init_pauli_sum.coefficients

    return pauli_sum


def exp_val(pauli_sum: PauliSum):
    """
    Returns the expectation value of pauli_sum in the |00...0> state.
    """
    
    num_terms = pauli_sum.S_z.shape[1]
    
    coefficients = pauli_sum.coefficients
    if coefficients is None:
        coefficients = np.ones(num_terms)
    
    overlapping = 1 - np.any(pauli_sum.S_x, 0)

    return np.sum(coefficients*overlapping)


def clifford_expectation(circuit: QuantumCircuit, hamiltonian: PauliSum, x):
    """
    Args:
        circuit: a Qiskit QuantumCircuit object representing an ansatz for 
            constructing a quantum state from the |0..0> state

        hamiltonian: a PauliSum object whose expectation value will be returned

        x: discrete parameters. Array of integers from 0 to 3 representing
            rotation angles 0, pi/2, pi, 3*pi/2 to be used for rotation gates
            in circuit.

    Returns:
        Expectation value of hamiltonian for state defined by circuit and x.
    """
    clifford_circuit = cliffordize(parse_circuit(circuit)[1], x)
    value = exp_val(stabilizer_evolve(hamiltonian, clifford_circuit,
                                        reverse=True))
    return value


def clifford_optimize(
    circuit: QuantumCircuit, hamiltonian: PauliSum, x0=None, maxiter=100, 
    max_evals=8000, random=False, callback=None
):
    """
    Performs a search for Clifford states that minimize the expectation value
    of hamiltionian.

    Args:
        circuit: List of gates, as outputted by cliffordize().

        hamiltonian: PauliSum object whose lowest eigenvalue we are after.

        x0: np.array of integers from 0 to 3 representing initial
            discrete parameters. Will be chosen randomly if argument is absent

        maxiter: Maximum number of iterations.

        max_evals: Maximum number of evalutions of the objective function.

        random: The parameter to update will be chosen randomly if
                this is set to True. Will update sequentially if False.
                If not specified, this will default to False.
    
    Returns:
        Qiskit VQEResult object
    """

    def _objective(x):
        clifford_circuit = cliffordize(parse_circuit(circuit)[1], x)
        value = exp_val(stabilizer_evolve(hamiltonian, clifford_circuit,
                                          reverse=True))
        return value

    jacobian = get_clifford_jacobian(_objective)


    # Select an initial point for the ansatzs' parameters
    num_parameters = circuit.num_parameters
    if x0 is None:
        x0 = np.random.randint(0, 4, (num_parameters))
    
    # Run optimization

    if random:
        parameter_index = np.random.randint(num_parameters)
    else:
        parameter_index = 0

    x = np.copy(x0)
    x_opt = np.copy(x0)

    function_value = _objective(x0)
    function_value_opt = function_value
    
    function_evaluations = 0
    iteration = 0
    unchanged_count = 0
    value_lowered = True
    while iteration <= maxiter and function_evaluations <= max_evals:
        
        for i in range(4):

            # evaluate function with new discrete parameters
            x_new = np.copy(x)
            x_new[parameter_index] = i
            candidate_value = _objective(x_new)
            function_evaluations += 1

            # update if we get a lower value than current value
            if candidate_value <= function_value:
                if candidate_value < function_value:
                    value_lowered = True
                else:
                    value_lowered = False
                x = x_new
                function_value = candidate_value
            
            # update if we get a globally lower value
            if candidate_value <= function_value_opt:
                x_opt = x.copy()
                function_value_opt = candidate_value

        if value_lowered == False:
            unchanged_count += 1
        else:
            unchanged_count = 0
        
        iteration += 1
        
        if callback is not None:
            callback([function_value, unchanged_count])
        
        if random:
            parameter_index = np.random.randint(num_parameters)
        else:
            parameter_index = (parameter_index + 1) % len(x)


        # Add random changes to avoid local minima
        if unchanged_count >= 100: # Can change from 100 to something else
            # replace_num = int(num_parameters*0.1)
            replace_num = 4 # Can change from 4 to something else
            replace_index = np.random.choice(
                range(num_parameters), replace_num, replace=False
            )
            x[replace_index] = np.random.randint(0, 4, replace_num)
            function_value = _objective(x)
            function_evaluations += 1
            # print('discrete parameters replaced')
            unchanged_count = 0

    if function_evaluations > max_evals:
        print(f'max_evals={max_evals} function evaluations reached')

    # Populate VQE result
    result = VQEResult()
    result.cost_function_evals = function_evaluations
    result.eigenvalue = function_value_opt
    result.optimal_parameters = x_opt
    return result


def get_clifford_jacobian(fun):
    """
    Returns a function jacobian() that calculates the Jacobian at a 
    Clifford state. This works for expectation values of parametrized
    states where the parameters are the angles of rotation gates and
    where the parameters at the point we want to find the Jacobian are
    all multiples of pi/2.

    The argument fun should be a function that takes as input an array of 
    integers from 0 to 3 representing discrete parameters for the Clifford 
    circuit.

    Note:
        see https://arxiv.org/abs/1903.12166
    """

    def jacobian(x):
        gradient = np.zeros(len(x))
        for index in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[index] = (x_plus[index] + 1) % 4
            x_minus[index] = (x_minus[index] - 1) % 4
            gradient[index] = (fun(x_plus) - fun(x_minus))/2
        return gradient

    return jacobian


def get_clifford_hessian(fun):
    """
    Returns a function hessian() that calculates the Hessian at a 
    Clifford state. This works for expectation values of parametrized
    states where the parameters are the angles of rotation gates and
    where the parameters at the point we want to find the Jacobian are
    all multiples of pi/2.

    The argument fun should be a function that takes as input an array of 
    integers from 0 to 3 representing discrete parameters for the Clifford 
    circuit.
    """

    def hessian(x):

        hess = np.zeros((len(x), len(x)))

        # off-diagonals
        for i in range(len(x)):
            for j in range(i):
                x_plus_plus = x.copy()
                x_plus_minus = x.copy()
                x_minus_plus = x.copy()
                x_minus_minus = x.copy()
                x_plus_plus[[i, j]] = [(x[i] + 1) % 4, (x[j] + 1) % 4]
                x_plus_minus[[i, j]] = [(x[i] + 1) % 4, (x[j] - 1) % 4]
                x_minus_plus[[i, j]] = [(x[i] - 1) % 4, (x[j] + 1) % 4]
                x_minus_minus[[i, j]] = [(x[i] - 1) % 4, (x[j] - 1) % 4]
                hess[i, j] = (fun(x_plus_plus) 
                              - fun(x_plus_minus) 
                              - fun(x_minus_plus) 
                              + fun(x_minus_minus))/4
        hess = hess + hess.T

        # diagonals
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] = (x[i] + 2) % 4
            hess[i, i] = (fun(x_plus) - fun(x))/2

        return hess

    return hessian




# def clifford_gradient_descent(
#     circuit: QuantumCircuit, hamiltonian: PauliSum, x0=None, jac=None,
#     maxiter=100, callback=None
# ):

#     def _objective(discrete_parameters):
#         return clifford_expectation(circuit, hamiltonian, discrete_parameters)

#     num_parameters = circuit.num_parameters
#     if x0 is None:
#         x0 = np.random.randint(0, 4, (num_parameters))

#     if jac is None:
#         jac = get_clifford_jacobian(_objective)

#     x = np.copy(x0)
#     x_opt = np.copy(x0)
#     function_value = _objective(x0)
#     function_value_opt = function_value
#     iteration = 0
#     while iteration <= maxiter:
        
#         gradient = jac(x)
#         change_index = np.argmax(np.abs(gradient))
#         change_sign = -np.sign(gradient[change_index])
#         print(f'Max derivative: {gradient[change_index]} at {change_index}', end='\r')

#         if change_sign == 0:
#             print('Found local minimum')
#             result = VQEResult()
#             result.eigenvalue = function_value
#             result.optimal_parameters = x
#             return

#         # evaluate function with new discrete parameters
#         x_new = np.copy(x)
#         x_new[change_index] = (x_new[change_index] + change_sign) % 4
#         candidate_value = _objective(x_new)
#         print(f'old value = {function_value} and new value = {candidate_value}')
        
#         x = x_new
#         function_value = candidate_value
#         iteration += 1
        
#         if callback is not None:
#             callback(function_value)

#     # Populate VQE result
#     result = VQEResult()
#     result.eigenvalue = function_value_opt
#     result.optimal_parameters = x_opt
#     return result


# def get_jacobian(fun):
#     """
#     Returns a function jacobian() that calculates the Jacobian of the
#     expectation value of a parametrized state. The parameters should
#     all be the angles of rotation gates. Therefore the argument fun
#     should be a function that takes as input an array of angles.
#     """

#     def jacobian(thetas):
#         gradient = np.zeros(len(thetas))
#         for index in range(len(thetas)):
#             thetas_plus = thetas.copy()
#             thetas_minus = thetas.copy()
#             thetas_plus[index] = thetas_plus[index] + np.pi/2
#             thetas_minus[index] = thetas_minus[index] - np.pi/2
#             gradient[index] = (fun(thetas_plus) - fun(thetas_minus))/2
#         return gradient

#     return jacobian
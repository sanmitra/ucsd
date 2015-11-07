from __future__ import division
from decimal    import Decimal
from functools  import lru_cache
from matplotlib import pyplot, patches
import numpy
import operator


class PairHMM:
    
    def __init__(self, states, symbols, transitional_probs, emission_probs):
        """
        TODO: add an option for start and end probability.
        """
        self.states = states
        self.symbols = symbols
        self.transitional_probs = transitional_probs
        self.emission_probs = emission_probs

    def simulate_data(self, N, L):
        """
        Generate @param{N} samples each of length @param{L}
        """
        state_sequences = []
        symbol_sequences = []
        for j in range(N):
            # Randomly choose start state.
            cur_state = numpy.random.choice(a=self.states)
            cur_symbol = numpy.random.choice(a=self.symbols, 
                                             p=self.emission_probs[self.states.index(cur_state)])

            state_sequence = cur_state
            symbol_sequence = cur_symbol

            # Randomly choose state to be transited from cur_state L-1 times based on the transition probabilities.
            for i in range(1, L):
                cur_state = numpy.random.choice(a=self.states,  
                                                p=self.transitional_probs[self.states.index(cur_state)])
                cur_symbol = numpy.random.choice(a=self.symbols, 
                                                 p=self.emission_probs[self.states.index(cur_state)])
                state_sequence += cur_state
                symbol_sequence += cur_symbol
            
            state_sequences.append(state_sequence)
            symbol_sequences.append(symbol_sequence)
            
        return (symbol_sequences, state_sequences)
    
    @lru_cache(maxsize=None)
    def get_most_probable_state_path(self, sequence):
        """
        Implements the Viterbi algorithm to get most probable state sequence path.
        """
        seq_length = len(sequence)
        no_states = len(self.states)
        V = self.get_zero_matrix(no_states, seq_length)
        path = self.get_zero_matrix(no_states, seq_length)
        for i in range(seq_length):
            for l in range(no_states):
                V[l][i] = self.emission_probs[l][self.symbols.index(sequence[i])]
                if i > 0:
                    max_index, max_value = max(enumerate([V[k][i-1] * self.transitional_probs[k][l] 
                                                          for k in range(no_states)
                                                         ]), 
                                               key=operator.itemgetter(1))
                    V[l][i] = V[l][i] * max_value
                    path[l][i] = max_index
        
        
        best_path_index = numpy.array(V)[:,-1].argmax()
        state_sequence = self.states[best_path_index]
        for i in reversed(range(1, seq_length)):
            best_path_index = path[best_path_index][i]
            state_sequence = self.states[best_path_index] + state_sequence
        return state_sequence

    @lru_cache(maxsize=None)
    def get_forward_probs(self, sequence):
        """
        This function implememts the forward algorithm.
        """
        seq_length = len(sequence)
        no_states = len(self.states)
        V = self.get_zero_matrix(no_states, seq_length)
        for i in range(seq_length):
            for l in range(no_states):
                V[l][i] = self.emission_probs[l][self.symbols.index(sequence[i])]
                # TODO : Delete this condition used for testing.
                if i == 0:
                    init_prob = [0.2, 0.8]
                    #V[l][i] = V[l][i] * init_prob[l]
                if i > 0:
                    V[l][i] = V[l][i] * sum([V[k][i-1] * self.transitional_probs[k][l] 
                                             for k in range(no_states)])
        return numpy.array(V)

    
    @lru_cache(maxsize=None)
    def get_backward_probs(self, sequence):
        """
        This function implememts the backward algorithm.
        """
        seq_length = len(sequence)
        no_states = len(self.states)
        V = self.get_zero_matrix(no_states, seq_length)
        for i in reversed(range(seq_length)):
            for k in range(no_states):
                if i == seq_length - 1:
                    V[k][i] = 1
                else:
                    V[k][i] = sum([V[l][i+1] * self.transitional_probs[k][l] 
                                   * self.emission_probs[l][self.symbols.index(sequence[i+1])]
                                   for l in range(no_states)])
        return numpy.array(V)
    
    
    def get_posterior_probability(self, sequence, observation_index, state_index):
        """
        Probability that observation X[observation_index] came from state[state_index] 
        in the given sequence.
        """
        return (self.get_forward_probs(sequence)[state_index][observation_index]
                * self.get_backward_probs(sequence)[state_index][observation_index]
                ) / self.get_probability_of_sequence(sequence)

    def get_probability_of_sequence(self, sequence):
        """
        Gets the probability of a sequence.
        """
        return sum(self.get_forward_probs(sequence)[:,-1])
    
    def get_cpg_posterior_probabilities(self, sequence):
        """
        Get posterior probabilities whether each base is in a CpG island.
        """
        return [sum([self.get_posterior_probability(sequence, i,  k) * (1 if "+" in self.states[k] else 0) 
                     for k in range(len(self.states))]) 
                for i in range(len(sequence))]


    def estimate_parameters(self, sequences, max_steps):
        """
        Implements the Baum Welch algorithm to estimate model parameters by learning 
        from the training sequences.
        """
        A = numpy.zeros([len(self.states), len(self.states)], dtype=object)
        E = numpy.zeros([len(self.states), len(self.symbols)], dtype=object)
        for step in range(max_steps):
            for sequence in sequences:
                # Clear the cache
                self.get_forward_probs.cache_clear()
                self.get_backward_probs.cache_clear()
                self.get_most_probable_state_path.cache_clear()

                fwd_probs = self.get_forward_probs(sequence)
                bwd_probs = self.get_backward_probs(sequence)
                seq_prob = self.get_probability_of_sequence(sequence)
                for k in range(len(self.states)):
                    for l in range(len(self.states)):
                        count = fwd_probs[k][len(sequence) - 1] * self.transitional_probs[k][l]
                        count += sum([(fwd_probs[k][i] * self.transitional_probs[k][l] * 
                                     self.emission_probs[l][self.symbols.index(sequence[i+1])] * bwd_probs[l][i+1])
                                     for i in range(len(sequence) - 1)])
                        A[k][l] += (count / seq_prob)
                
                for k in range(len(self.states)):
                    for b in range(len(self.symbols)):
                        count = sum([(fwd_probs[k][i] * bwd_probs[k][i]) if sequence[i] == self.symbols[b] else 0
                                    for i in range(len(sequence))])
                        E[k][b] += (count / seq_prob)
            
            # Update the transition and emission probabilities 
            # with newly estimated counts of transitions and emissions
            for k in range(len(self.states)):
                for l in range(len(self.states)):
                    self.transitional_probs[k][l] = A[k][l]/sum(A[k])
            
            for k in range(len(self.states)):
                for b in range(len(self.symbols)):
                    self.emission_probs[k][b] = E[k][b]/sum(E[k])

    @staticmethod
    def get_zero_matrix(M, N):
        return [[0 for j in range(N)] for i in range(M)]


def test_dishonest_casino_model():
    states = ["F", "L"]
    symbols = ['1', '2', '3', '4', '5', '6']
    transitional_probs = [[0.95, 0.05], 
                          [0.1, 0.9]]
    emission_probs = [[1/6,1/6,1/6,1/6,1/6,1/6], 
                      [1/10,1/10,1/10,1/10,1/10,1/2]]
    model = PairHMM(states, symbols, transitional_probs, emission_probs)
    sequence = "".join(['315116246446644245311321631164152133625144543631656626566666', 
                        '651166453132651245636664631636663162326455236266666625151631',
                        '222555441666566563564324364131513465146353411126414626253356',
                        '366163666466232534413661661163252562462255265252266435353336',
                        '233121625364414432335163243633665562466662632666612355245242'
                         ])
    state_path = model.get_most_probable_state_path(sequence)
    for i in range(5):
        print(sequence[i*60 :i*60+60])
        print(state_path[i*60 : i*60+60])
        print("\n")

    x_coordinates = [i+1 for i in range(len(sequence))]
    y_coordinates = [model.get_posterior_probability(sequence, i, 0) for i in range(len(sequence))]
    x_coordinates.insert(0, 0)
    y_coordinates.insert(0, 1)
    pyplot.plot(x_coordinates, y_coordinates)
    pyplot.show()

def test_cpg_model(model, symbol_sequences, state_sequences, file_name):
    # For each simulated data we calculate its likelihood
    # get the optimal state sequence and also posterior probability 
    # whether each base is in a CpG island.
    for i in range(len(symbol_sequences)):
        viterbi_state_sequence = model.get_most_probable_state_path(symbol_sequences[i])
        viterbi_cpg_sequence = "".join(filter(lambda x: x == "+" or x == "-", viterbi_state_sequence))

        print("DNA sequence:           " + symbol_sequences[i])
        print("True State sequence:    " + state_sequences[i])
        print("Viterbi State sequence: " + viterbi_state_sequence)
        print("True CpG Sequence:      " + "".join(filter(lambda x: x == "+" or x == "-", state_sequences[i])))
        print("Viterbi CpG Sequence:   " + viterbi_cpg_sequence)
        print("Likelihood:             " + str(model.get_probability_of_sequence(symbol_sequences[i])))
        print("\n")
        
        # Create a new figure.
        pyplot.figure()
        # Plot the posterior probabilities whether each base is in a CpG island.
        y_coordinates = model.get_cpg_posterior_probabilities(symbol_sequences[i])
        x_coordinates = [k + 1 for k in range(len(symbol_sequences[i]))]
        pyplot.plot(x_coordinates, y_coordinates)
        pyplot.ylabel('Posterior Probability')
        pyplot.xlabel('Sequence Length')

        # Plot the shades depicting the cpg islands decoded by Viterbi algorithm
        viterbi_state_sequence = model.get_most_probable_state_path(symbol_sequences[i])
        viterbi_cpg_sequence = "".join(filter(lambda x: x == "+" or x == "-", viterbi_state_sequence))
        start = None
        end = None
        currentAxis = pyplot.gca()
        for j in range(len(viterbi_cpg_sequence)):
            if not start and viterbi_cpg_sequence[j] == "+":
                start = j
            if start and viterbi_cpg_sequence[j] == "-":
                end = j
                currentAxis.add_patch(patches.Rectangle((start, 0), (end-start), 1, facecolor='lightgrey'))
                start = None
                end = None
        pyplot.savefig(file_name + "_posterior_viterbi_" + str(i+1) + ".png")

def test(N, L, file_name):
    """
    Run a test case for N samples of length L
    """
    states = ["A+", "C+", "G+", "T+",
              "A-", "C-", "G-", "T-"]
    symbols = ["A", "C", "G", "T"]
    transition_probabilities = [
                                [Decimal('0.144'), Decimal('0.2192'), Decimal('0.3408'), Decimal('0.096'), Decimal('0.036'), Decimal('0.0548'), Decimal('0.0852'), Decimal('0.024')],
                                [Decimal('0.136'), Decimal('0.2944'), Decimal('0.2192'), Decimal('0.1504'), Decimal('0.034'), Decimal('0.0736'), Decimal('0.0548'), Decimal('0.0376')],
                                [Decimal('0.1288'), Decimal('0.2712'), Decimal('0.300'), Decimal('0.100'), Decimal('0.0322'), Decimal('0.0678'), Decimal('0.075'), Decimal('0.025')],
                                [Decimal('0.0632'), Decimal('0.284'), Decimal('0.3072'), Decimal('0.1456'), Decimal('0.0158'), Decimal('0.071'), Decimal('0.0768'), Decimal('0.0364')],
                                [Decimal('0.045'), Decimal('0.03075'), Decimal('0.04275'), Decimal('0.0315'), Decimal('0.255'), Decimal('0.17425'), Decimal('0.24225'), Decimal('0.1785')],
                                [Decimal('0.0483'), Decimal('0.0447'), Decimal('0.0117'), Decimal('0.0453'), Decimal('0.2737'), Decimal('0.2533'), Decimal('0.0663'), Decimal('0.2567')],
                                [Decimal('0.0372'), Decimal('0.0369'), Decimal('0.0447'), Decimal('0.0312'), Decimal('0.2108'), Decimal('0.2091'), Decimal('0.2533'), Decimal('0.1768')],
                                [Decimal('0.02655'), Decimal('0.03585'), Decimal('0.0438'), Decimal('0.0438'), Decimal('0.15045'), Decimal('0.20315'), Decimal('0.2482'), Decimal('0.2482')],
                                ]
    emission_probabilities = [[1,0,0,0],
                              [0,1,0,0],
                              [0,0,1,0],
                              [0,0,0,1],
                              [1,0,0,0],
                              [0,1,0,0],
                              [0,0,1,0],
                              [0,0,0,1]]
    
    print("Testing the model assuming that we know true parameters")
    model = PairHMM(states, symbols, transition_probabilities, emission_probabilities)
    # Assuming the above arbitrarily chosen model parameters as true simulate the data
    # according to the model parameters.
    symbol_sequences, state_sequences = model.simulate_data(N, L)
    test_cpg_model(model, symbol_sequences, state_sequences, file_name+"_true_")
    
    
    print("\nEstimating the parameters based on the simulated data and test the model with new parameters.")
    # Arbitrary parameters
    arb_transition_probabilities = [
        [Decimal(str(i)) for i in numpy.random.dirichlet(numpy.ones(8),size=1)[0]],
        [Decimal(str(i)) for i in numpy.random.dirichlet(numpy.ones(8),size=1)[0]],
        [Decimal(str(i)) for i in numpy.random.dirichlet(numpy.ones(8),size=1)[0]],
        [Decimal(str(i)) for i in numpy.random.dirichlet(numpy.ones(8),size=1)[0]],
        [Decimal(str(i)) for i in numpy.random.dirichlet(numpy.ones(8),size=1)[0]],
        [Decimal(str(i)) for i in numpy.random.dirichlet(numpy.ones(8),size=1)[0]],
        [Decimal(str(i)) for i in numpy.random.dirichlet(numpy.ones(8),size=1)[0]],
        [Decimal(str(i)) for i in numpy.random.dirichlet(numpy.ones(8),size=1)[0]],
    ]
    train_model = PairHMM(states, symbols, arb_transition_probabilities, emission_probabilities)
    train_model.estimate_parameters(symbol_sequences, 10)
    test_cpg_model(train_model, symbol_sequences, state_sequences, file_name+"_estimated_")
    
if __name__ == "__main__":
    print("START Test Case 1\n")
    test(1, 1000, "test_case1")
    print("END Test Case 1\n")

    print("START Test Case 2\n")
    test(10, 1000, "test_case2")
    print("END Test Case 2\n")

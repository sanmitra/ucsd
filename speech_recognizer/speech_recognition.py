from __future__ import division
from decimal    import Decimal
from matplotlib import pyplot
import math
import numpy
import string

#####################################################################################################

# File names
TRANSITION_MATRIX_FILE = "hw6_transitionMatrix.txt"
INITAL_STATE_DISTRIBUTION_FILE = "hw6_initialStateDistribution.txt"
EMISSION_MATRIX_FILE = "hw6_emissionMatrix.txt"
OBSERVATIONS_MATRIX_FILE = "hw6_observations.txt"

#####################################################################################################

class HMM:
    def __init__(self, states, symbols, initial_probabilities, transitional_probs, emission_probs):
        """
        Initial HMM object with all the model parameters.
        Note: All the probabilities should be in log probabilities.
        """
        self.states = states
        self.symbols = symbols
        self.initial_probabilities = numpy.array(initial_probabilities)
        self.transitional_probs = numpy.array(transitional_probs)
        self.emission_probs = numpy.array(emission_probs)

    @staticmethod
    def get_zero_matrix(M, N):
        return [[0 for j in range(N)] for i in range(M)]
    
    def get_most_probable_state_path(self, sequence):
        """
        Implements the Viterbi algorithm to get most probable state sequence path.
        """
        seq_length = len(sequence)
        no_states = len(self.states)
        V = numpy.zeros((no_states, seq_length))
        path = numpy.zeros((no_states, seq_length), dtype=int)
        for i in range(seq_length):
            for l in range(no_states):
                V[l][i] = self.emission_probs[l][self.symbols.index(sequence[i])]
                if i == 0:
                    V[l][i] = V[l][i]  + self.initial_probabilities[l]
                if i > 0:
                    arr = V[:,i-1] + self.transitional_probs[:,l]
                    V[l][i] = V[l][i] + arr.max()
                    path[l][i] = arr.argmax()

        best_path_index = V[:,-1].argmax()
        state_sequence = self.states[best_path_index]
        for i in reversed(range(1, seq_length)):
            best_path_index = path[best_path_index][i]
            state_sequence = self.states[best_path_index] + state_sequence
        return state_sequence

#####################################################################################################

def parse_inputs():
    """
    This file parses the below input files to get HMM parameters for speech recognition.
    hw6_transitionMatrix.txt
    hw6_initialStateDistribution.txt
    hw6_emissionMatrix.txt
    hw6_observations.txt
    """
    dec2log = lambda dec : math.log(Decimal(dec))
    with open(TRANSITION_MATRIX_FILE, "r") as fp1:
        transitional_probs = list(map(lambda line: list(map(dec2log, filter(None, line.replace('\n', '').split(" ")))), fp1.readlines()))
    with open(INITAL_STATE_DISTRIBUTION_FILE, "r") as fp2:
        initial_probabilities = list(map(lambda line: dec2log(line.replace('\n', '')), fp2.readlines())) 
    with open(EMISSION_MATRIX_FILE, "r") as fp3:
        emission_probs = list(map(lambda line: list(map(dec2log, filter(None, line.replace('\n', '').split('\t')))), fp3.readlines()))
    with open(OBSERVATIONS_MATRIX_FILE, "r") as fp4:
        observed_sequence = list(filter(None, fp4.readlines()[0].replace('\n', '').split(" ")))
    return transitional_probs, initial_probabilities, emission_probs, observed_sequence
    
    
def test_speech_recognizer():
    """
    Test the speech recognizer by decoding a meaningful message 
    from real valued measurements of acoustic waveforms. 
    """
    states = list(string.ascii_lowercase)
    symbols = ['0', '1']
    transitional_probs, initial_probabilities, emission_probs, observed_sequence = parse_inputs()
    model = HMM(states, symbols, initial_probabilities, transitional_probs, emission_probs)
    state_path = model.get_most_probable_state_path("".join(observed_sequence))
    
    # Remove the repetitive letters from state sequence obtained to get a meaningful message.
    sentence = state_path[0]
    Y = numpy.zeros(len(state_path))
    Y[0] = states.index(state_path[0]) + 1
    for i in range(1, len(state_path)):
        if (state_path[i] != state_path[i-1]):
            sentence = sentence + state_path[i]
            Y[i] = states.index(state_path[i]) + 1
        else:
            Y[i] = Y[i-1]
    
    # Write the complete sequence.
    with open("complete_state_sequence_output.txt", "w") as fp:
        fp.write(state_path)
    
    # Write the meaningful sentence i.e the one with non repetitive letters.
    with open("sentence_output.txt", "w") as fp:
        fp.write(sentence)
    
    # Plot the graph with probable state vs observed sequence.
    X = list(range(1, len(state_path) + 1))
    pyplot.plot(X, Y)
    pyplot.yticks(list(range(1, len(states) + 1)), states)
    pyplot.show()
    
#####################################################################################################
   
if __name__ == "__main__":
    test_speech_recognizer()

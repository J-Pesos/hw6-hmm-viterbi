import copy
import numpy as np

class ViterbiAlgorithm:
    """Implements the Viterbi Algorithm on a hmm_object, which is instantiated as a class attribute.
    """    

    def __init__(self, hmm_object):
        """Initialization of the Vertibi Algorithm class.

        Args:
            hmm_object (object): Hidden Markov Model object containing state names, index state name dictionaries, and probabilities.
        """              
        self.hmm_object = hmm_object

    def best_hidden_state_sequence(self, decode_observation_states: np.ndarray) -> np.ndarray:
        """Determines the most probable hidden state sequence given a sequence of observation states.

        Args:
            decode_observation_states (np.ndarray): Sequence of the observation states.

        Returns:
            np.ndarray: Most probable hidden state sequence.
        """        
        
        # Initialize path (i.e., np.arrays) to store the hidden sequence states returning the maximum probability
        path = np.zeros((len(decode_observation_states), 
                         len(self.hmm_object.hidden_states)))
        path[0,:] = [hidden_state_index for hidden_state_index in range(len(self.hmm_object.hidden_states))]

        best_path = np.zeros(len(decode_observation_states))
        
        # Redefine varaibles from hmm_object to easily reference.
        observation_states_dict = self.hmm_object.observation_states_dict
        hidden_states = self.hmm_object.hidden_states
        prior_p = self.hmm_object.prior_probabilities
        transition_p = self.hmm_object.transition_probabilities
        emission_p = self.hmm_object.emission_probabilities
        
        # Compute initial delta:
        # 1. Calculate the product of the prior and emission probabilities. This will be used to decode the first observation state.
        delta = np.multiply(prior_p, np.transpose(emission_p[:,
                                                             observation_states_dict.get(decode_observation_states[0])]))

        # 2. Scale      
        # Delta values should sum to 1.
        delta = delta / np.sum(delta)
        best_path[0] = path[0][np.argmax(delta)]

        # For each observation state to decode, select the hidden state sequence with the highest probability (i.e., Viterbi trellis)
        for trellis_node in range(1, len(decode_observation_states)):
            # Calculate the product of the delta and transition probabilities.
            product_of_delta_and_transition =  np.multiply(delta, transition_p.transpose())

            # Calculate the product of delta and transition product and emission probability of the observation at the current state of trellis path.
            # Important to note that rows are hidden states and columns are observed states in emission probabilities.
            product_of_delta_and_transition_emission = np.multiply(product_of_delta_and_transition,
                                                                   emission_p[:, observation_states_dict.get(decode_observation_states[trellis_node])])
            
            # Maximize product_of_delta_and_transition_emission by getting max for each column, which is what each hidden and observed state corresponds to.
            # Also, transpose so that observed states are rows and hidden states are columns.
            maxp_hidden_state = product_of_delta_and_transition_emission.max(axis=1).transpose()

            # Scale probabilites.
            scaled_maxp = maxp_hidden_state / np.sum(maxp_hidden_state)

            # Track indices of observed states dependent on max value. Needed to decode hidden states.
            observed_index = product_of_delta_and_transition_emission.argmax(axis = 1)

            # Add observed state indices to path.
            path[trellis_node] = observed_index

            # Get the next observed state based on max probability using our delta.
            best_path[trellis_node - 1] = path[trellis_node - 1][np.argmax(scaled_maxp)]

            # Update delta.
            delta = np.multiply(prior_p, np.transpose(emission_p[:,
                                                             observation_states_dict.get(decode_observation_states[trellis_node])]))

        best_hidden_state_path = np.array([hidden_states[np.int32(i)] for i in best_path])
        return best_hidden_state_path
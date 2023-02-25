import numpy as np
class HiddenMarkovModel:
    """Instantiate Hideen Markov Model containing observation and hidden states, and prior/transition/emission probabilities. Used to calculate most probable hidden states.
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_probabilities: np.ndarray, transition_probabilities: np.ndarray, emission_probabilities: np.ndarray):
        """Initializes HiddenMarkovModel object.

        Args:
            observation_states (np.ndarray): 1D array of all possible observation states.
            hidden_states (np.ndarray): 1D array of all possible observation states.
            prior_probabilities (np.ndarray): 1D array of prior probabilities of each observed state.
            transition_probabilities (np.ndarray): 2D array of transition probabilites of hidden states changing from one to another. Rows are observed states and columns are hidden states.
            emission_probabilities (np.ndarray): 2D array of emission probabilities of hidden states emitting observed states. Rows are hidden states and columns are observed states.
        """             
        self.observation_states = observation_states
        self.observation_states_dict = {observation_state: observation_state_index \
                                  for observation_state_index, observation_state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {hidden_state_index: hidden_state \
                                   for hidden_state_index, hidden_state in enumerate(list(self.hidden_states))}
        

        self.prior_probabilities= prior_probabilities
        self.transition_probabilities = transition_probabilities
        self.emission_probabilities = emission_probabilities
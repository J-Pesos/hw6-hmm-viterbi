"""
UCSF BMI203: Biocomputing Algorithms
Author: Joaquín Magaña
Date: 2/24/23
Program: Test Cases
Description: Pytest for user test cases.
"""
import pytest
import numpy as np
from models.hmm import HiddenMarkovModel
from models.decoders import ViterbiAlgorithm


def test_use_case_lecture():
    """Test case to evaluate HMM predictions of a graduate student's dedication based on their rotation lab's funding source.
    """
    # index annotation observation_states=[i,j]    
    observation_states = ['committed','ambivalent'] # A graduate student's dedication to their rotation lab
    
    # index annotation hidden_states=[i,j]
    hidden_states = ['R01','R21'] # The NIH funding source of the graduate student's rotation project 

    # PONDERING QUESTION: How would a user define/compute their own HMM instantiation inputs to decode the hidden states for their use case observations?
    use_case_one_data = np.load('./data/UserCase-Lecture.npz')

    # Instantiate submodule class models.HiddenMarkovModel with
    # observation and hidden states and prior, transition, and emission probabilities.
    use_case_one_hmm = HiddenMarkovModel(observation_states,
                                         hidden_states,
                      use_case_one_data['prior_probabilities'], # prior probabilities of hidden states in the order specified in the hidden_states list
                      use_case_one_data['transition_probabilities'], # transition_probabilities[:,hidden_states[i]]
                      use_case_one_data['emission_probabilities']) # emission_probabilities[hidden_states[i],:][:,observation_states[j]]
    
    # Instantiate submodule class models.ViterbiAlgorithm using the use case one HMM 
    use_case_one_viterbi = ViterbiAlgorithm(use_case_one_hmm)

     # Check if use case one HiddenMarkovAlgorithm instance is inherited in the subsequent ViterbiAlgorithm instance
    assert use_case_one_viterbi.hmm_object.observation_states == use_case_one_hmm.observation_states
    assert use_case_one_viterbi.hmm_object.hidden_states == use_case_one_hmm.hidden_states

    assert np.allclose(use_case_one_viterbi.hmm_object.prior_probabilities, use_case_one_hmm.prior_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.transition_probabilities, use_case_one_hmm.transition_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.emission_probabilities, use_case_one_hmm.emission_probabilities)

    # Check HMM dimensions and ViterbiAlgorithm
    # Both states should have the same dimensions. Additionally, emission and transition probabilities should have the same dimensions.
    assert np.shape(use_case_one_viterbi.hmm_object.observation_states) == np.shape(use_case_one_viterbi.hmm_object.hidden_states)
    assert np.shape(use_case_one_viterbi.hmm_object.emission_probabilities) == np.shape(use_case_one_viterbi.hmm_object.transition_probabilities)
    
    # Find the best hidden state path for our observation states
    use_case_decoded_hidden_states = use_case_one_viterbi.best_hidden_state_sequence(use_case_one_data['observation_states'])
    assert np.alltrue(use_case_decoded_hidden_states == use_case_one_data['hidden_states'])


def test_user_case_one():
    """Test case to evaluate HMM predictions of punctuality (late/on-time) depending on presence of traffic.
    """
    # index annotation observation_states=[i,j]    
    observation_states = ['on-time','late'] 

    # index annotation hidden_states=[i,j]
    hidden_states = ['no-traffic','traffic']

    # PONDERING QUESTION: How would a user define/compute their own HMM instantiation inputs to decode the hidden states for their use case observations?
    use_case_one_data = np.load('./data/UserCase-One.npz')

    # Instantiate submodule class models.HiddenMarkovModel with
    # observation and hidden states and prior, transition, and emission probabilities.
    use_case_one_hmm = HiddenMarkovModel(observation_states,
                                         hidden_states,
                      use_case_one_data['prior_probabilities'], # prior probabilities of hidden states in the order specified in the hidden_states list
                      use_case_one_data['transition_probabilities'], # transition_probabilities[:,hidden_states[i]]
                      use_case_one_data['emission_probabilities']) # emission_probabilities[hidden_states[i],:][:,observation_states[j]]
    
    # Instantiate submodule class models.ViterbiAlgorithm using the use case one HMM 
    use_case_one_viterbi = ViterbiAlgorithm(use_case_one_hmm)

     # Check if use case one HiddenMarkovAlgorithm instance is inherited in the subsequent ViterbiAlgorithm instance
    assert use_case_one_viterbi.hmm_object.observation_states == use_case_one_hmm.observation_states
    assert use_case_one_viterbi.hmm_object.hidden_states == use_case_one_hmm.hidden_states

    assert np.allclose(use_case_one_viterbi.hmm_object.prior_probabilities, use_case_one_hmm.prior_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.transition_probabilities, use_case_one_hmm.transition_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.emission_probabilities, use_case_one_hmm.emission_probabilities)

    # Check HMM dimensions and ViterbiAlgorithm
    # Both states should have the same dimensions. Additionally, emission and transition probabilities should have the same dimensions.
    assert np.shape(use_case_one_viterbi.hmm_object.observation_states) == np.shape(use_case_one_viterbi.hmm_object.hidden_states)
    assert np.shape(use_case_one_viterbi.hmm_object.emission_probabilities) == np.shape(use_case_one_viterbi.hmm_object.transition_probabilities)
    
    # Find the best hidden state path for our observation states
    use_case_decoded_hidden_states = use_case_one_viterbi.best_hidden_state_sequence(use_case_one_data['observation_states'])
    hid_state_sequence = np.array(['no-traffic', 'no-traffic', 'traffic', 'traffic', 'traffic', 'no-traffic'])

    assert np.alltrue(use_case_decoded_hidden_states == hid_state_sequence)


def test_user_case_two():
    """Test case to evaluate HMM predictions of a dog being prone to anxiety should they have been adopted through a shelter.
    """
    observation_states = ['anxiety prone', 'not prone to anxiety']
    hidden_states = ['adopted through shelter', 'not adopted']

    prior_p = np.array([0.3, 0.7])
    transition_p = np.array([[0.25, 0.75],
                             [0.45, 0.55]])
    emission_p = np.array([[0.65, 0.35],
                          [0.15, 0.85]])
    
    obs_state_sequence = ['not prone to anxiety', 'anxiety prone', 'not prone to anxiety', 'not prone to anxiety', 'anxiety prone']
    hid_state_sequence = np.array(['not adopted', 'adopted through shelter', 'adopted through shelter', 'not adopted', 'adopted through shelter'])

    hmm_dog_anxiety = HiddenMarkovModel(observation_states = observation_states,
                                        hidden_states = hidden_states,
                                        prior_probabilities = prior_p,
                                        transition_probabilities = transition_p,
                                        emission_probabilities = emission_p)
    
    dog_anxiety_viterbi = ViterbiAlgorithm(hmm_dog_anxiety)
    
    viterbi_sequence = dog_anxiety_viterbi.best_hidden_state_sequence(obs_state_sequence)

    assert np.alltrue(hid_state_sequence == viterbi_sequence)


def test_user_case_three():
    """Test case to evaluate HMM predictions of a videogame ranking based on primary controller type used that season.
    """
    observation_states = ['Gold or above', 'below Gold']
    hidden_states = ['fight stick', 'gamepad']

    prior_p = np.array([0.55, 0.45])
    transition_p = np.array([[0.6, 0.4],
                             [0.35, 0.65]])
    emission_p = np.array([[0.5, 0.5],
                          [0.35, 0.65]])
    
    obs_state_sequence = ['below Gold', 'below Gold', 'Gold or above', 'below Gold', 'Gold or above']
    hid_state_sequence = np.array(['gamepad', 'fight stick', 'fight stick', 'fight stick', 'fight stick'])

    hmm_rank_controller = HiddenMarkovModel(observation_states = observation_states,
                                            hidden_states = hidden_states,
                                            prior_probabilities = prior_p,
                                            transition_probabilities = transition_p,
                                            emission_probabilities = emission_p)
    
    rank_controller_viterbi = ViterbiAlgorithm(hmm_rank_controller)
    
    viterbi_sequence = rank_controller_viterbi.best_hidden_state_sequence(obs_state_sequence)

    assert np.alltrue(hid_state_sequence == viterbi_sequence)
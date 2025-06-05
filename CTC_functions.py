""" 
Functions list:

 - edit_distance(word1: str, word2: str) -> int

 - label_error_rate(predicted: list, target: list) -> float

 - modified_label_sequence(word: str) -> str

 - compute_forward_values(probs: np.ndarray, target_sequence: list[int]) -> tuple[np.ndarray, np.ndarray]
"""

import numpy as np

def edit_distance(word1: str, word2: str) -> int:
    """
    Compute the edit distance between two strings.

    The edit distance between two strings is the minimum number of operations
    required to transform one string into the other. The permitted operations are:
    - Delete a character
    - Insert a character
    - Replace a character

    Parameters:
    - word1 (str): The source string
    - word2 (str): The target string

    Returns:
    - int: The edit distance between word1 and word2
    """

    n1 = len(word1)
    n2 = len(word2)

    # DP[i][j] = edit distance between word1[i:] and word2[j:]
    DP = [[0 for _ in range(n2+1)] for _ in range(n1+1)]

    # Base Case (1): word1[n1:] = '' (last row of the DP matrix)
    for col in reversed(range(n2+1)):
        DP[-1][col] = n2 - col

    # Base Case (2): word2[n2:] = '' (last column of the DP matrix)
    for row in reversed(range(n1+1)):
        DP[row][-1] = n1 - row

    # Iterative Case: 
    for row in reversed(range(n1)):
        for col in reversed(range(n2)):
            if word1[row] == word2[col]:
                DP[row][col] = DP[row+1][col+1]
            else:
                insert_op  = 1 + DP[row][col+1] 
                delete_op  = 1 + DP[row+1][col]
                replace_op = 1 + DP[row+1][col+1]
                DP[row][col] = min(insert_op, delete_op, replace_op)

    return DP[0][0]


def label_error_rate(predicted: list, target: list) -> float:
    """
    Compute the Label Error Rate (LER) between two sequences:
        LER(h,S) = 1/|S| sum_{(x,z) in S} ED(h(x),z)/|z|

    Args:
        - predicted (list): The predicted sequence of labels.
        - target (list): The ground truth sequence of labels.

    Returns:
        - float: The label error rate as a proportion.
    """
    assert len(predicted) == len(target), "Error: The given lists have different size" 
    assert len(predicted)*len(target) != 0, "Error: One of the given lists is empty"
    assert all(word != '' for word in target), "Error: All elements target must be non-empty strings"
    
    n = len(predicted)

    # Compute the edit distance between the predicted and target sequences
    distance = sum([edit_distance(p, t)/len(t) for p, t in zip(predicted, target)])

    # Compute the label error rate
    ler = distance / len(target)

    return ler

def modified_label_sequence(word: str) -> str:
    """
    Modify a sequence of labels by adding a blank space at the beginning, at the end,
    and between each character.

    Args:
        - word (str): The input sequence of labels.
    
    Returns:
        - str: The modified sequence with '-', that represent a blank spaces added.
    """
    # Add '-' between characters, then prepend and append '-'
    return '-' + '-'.join(word) + '-'
    
def compute_forward_values(probs: np.ndarray, target_sequence: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the forward values for the CTC loss using dynamic programming.

    Args:
        - probs : A (T, S) array where probs[t,s] is the probability of the t-th time step being the s-th label.
        - target_sequence : The target sequence of labels.

    Returns:
        - alpha: A (T, len(target_sequence)) array of forward values.
        - C: A length-T array used to normalize alpha at each time step.
    """
    assert len(target_sequence) <= probs.shape[1], "Error: Target sequence is too long for the probability matrix"

    T, _ = probs.shape  # Get the number of time steps and states

    len_target = len(target_sequence)

    # Initialize alpha and C
    alpha = np.zeros((T, len_target))
    C = np.zeros(T)

    # Base cases
    if len_target >= 2:
        alpha[0,1] = probs[0,target_sequence[0]]
        C[0] = alpha[0,0] + alpha[0,1]
    else:
        C[0] = alpha[0,0]

    # Recursion
    for t in range(1, T):
        for s in range(len_target): 
            alpha[t,s] += alpha[t-1, s]
            if s >= 1:
                alpha[t,s] += alpha[t-1, s-1]
            if s >= 2 and target_sequence[s] != target_sequence[s-2]:
                alpha[t,s] += alpha[t-1, s-2]
            alpha[t,s] *= probs[t, target_sequence[s]]
            C[t] += alpha[t,s]
        
        # Rescale row
        if C[t] != 0:
            alpha[t,:] /= C[t]

    return alpha, C
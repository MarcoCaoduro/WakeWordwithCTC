""" 
Functions index:

 - edit_distance(word1: str, word2: str) -> int:

 - label_error_rate(predicted: list, target: list) -> float:
"""

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
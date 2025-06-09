from typing import Union

def alphabet_coding(alphabet: list[str], sequence: Union[list[str], str]) -> list[int]:
    """
    Encode a sequence of characters using a custom alphabet.

    In the coding, the number `0` is reserved for a special 'blank' symbol,
    so each character in the alphabet is assigned a unique integer starting from 1.

    Args:
        alphabet (list[str]): The list of characters to encode. Each character gets an integer code.
        sequence (Union[list[str], str]): The sequence of characters to encode. Can be a list or a string.

    Returns:
        list[int]: The encoded sequence where each character is replaced by its corresponding integer code.
    """
    # Assign a unique integer to each character in the alphabet, starting from 1
    coding_f = {key: i + 1 for i, key in enumerate(alphabet)}

    # Map each element in the sequence to its corresponding code
    return [coding_f[element] for element in sequence]


def alphabet_decoding(alphabet: list[str], sequence: list[int]) -> str:
    """
    Decode a sequence of integers back into a string using a custom alphabet.

    Assumes that the encoding used integer values starting from 1 for the first
    character in the alphabet. The value 0 is assumed to be reserved for 'blank'
    and is not decoded here.

    Args:
        alphabet (list[str]): The original list of characters used for encoding.
        sequence (list[int]): The encoded sequence of integers to decode.

    Returns:
        str: The decoded string formed by mapping integers back to characters.
    """

    # Build a dictionary that maps each integer (starting from 1) to the corresponding character
    decoding_f = {i + 1: key for i, key in enumerate(alphabet)}

    # Convert the list of integers into a string by mapping each integer to its corresponding character
    return ''.join([decoding_f[element] for element in sequence])
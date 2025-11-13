import os
import argparse
import numpy as np


def int2str(num: int, num_digits: int) -> str:
    """
    Converts an integer to a string with:
        1. each digit separated by a whitespace
        2. padded with zeros to have always num_digits
        3. reversed order to have causual-friendly structure for Transformers
    """
    return " ".join(str(num)[::-1].rjust(num_digits, "0"))


def check_for_copy(line: str, all_lines_set: set[str]):
    """
    Checks if the line is not already within the all_lines_set (highly efficient, use a set!).
    """
    return line not in all_lines_set


def generate_data(
    digits_in_operands: int,
    num_samples: int,
    previous_examples: set[str],
) -> list[str]:
    maximum_value = int("9" * digits_in_operands)
    all_lines = []
    while len(all_lines) < num_samples:
        a, b = np.random.randint(1, maximum_value + 1, size=(2,))
        result = a * b
        a_str = int2str(a, digits_in_operands)
        b_str = int2str(b, digits_in_operands)
        res_str = int2str(result, 2 * digits_in_operands)

        line = f"{a_str} * {b_str}### {res_str}"
        if check_for_copy(line, previous_examples):
            all_lines.append(line + "\n")
            previous_examples.update(line)
    return all_lines


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_digits",
        type=int,
        required=True,
        nargs="+",
        help="Space separated list of numbers of digits in each operand (e.g., for 4-digit multiplication, use 4).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        required=True,
        help="Number of total multiplication samples to generate.",
    )
    parser.add_argument(
        "--fname",
        type=str,
        required=True,
        help="Base filename for the generated dataset (without extension).",
    )
    parser.add_argument(
        "--previous_datasets",
        type=str,
        required=False,
        nargs="+",
        help="Space separated list of file names to make sure there is no overlap between train, val and test set examples",
    )
    args = parser.parse_args()

    num_digits = args.num_digits
    num_samples = args.num_samples

    path = os.path.join("data", f"{num_digits}_by_{num_digits}")
    os.makedirs(path, exist_ok=True)

    previous_examples = set()
    if args.previous_datasets:
        for fname in args.previous_datasets:
            with open(os.path.join(path, fname + ".txt"), "r") as f:
                previous_examples.update(f.readlines())

    all_lines = []
    for digit_length in num_digits:
        all_lines += generate_data(digit_length, num_samples, previous_examples)
        previous_examples.update(all_lines)


    with open(os.path.join(path, f"{args.fname}.txt"), "w") as f:
        f.writelines(all_lines)

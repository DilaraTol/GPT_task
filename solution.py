import numpy as np
from io import StringIO


def extract_features(samples: np.ndarray):
    """
    samples: np.ndarray of shape (N, 100, 3)
    Returns: list of lists, each inner list has length F <= 20
    """
    # REPLACE WITH YOUR CODE
    features = []
    for sample in samples:
        feature = []
        # sample shape: (100, 3)
        for i in range(0,100,10):
            summ=0
            for j in range(0,10):
                for k in range(0,3):
                    summ += sample[j+i][k]
            feature.append(summ)
        feature.append(sample.mean())
        feature.append(sample.std())
        features.append(feature)
    return features


def get_total_input(text_lines):
    records = text_lines.strip().split("\n\n")
    N = int(records[0])
    records = records[1:]

    samples = []
    for i in range(N):
        # Each sample has 100 lines with 3 integers each
        sample_lines = records[i].strip().split("\n")
        if len(sample_lines) != 100:
            raise ValueError(f"Expected 100 lines for sample {i}, got {len(sample_lines)}")
        sample = np.loadtxt(StringIO("\n".join(sample_lines))).astype(np.int8)
        samples.append(sample)
    return np.array(samples)


def float_list_to_str(float_list):
    string_list = [f"{num:.6f}" for num in float_list]
    return ' '.join(string_list)


def generate_total_output(samples):
    features = extract_features(samples)
    features_string_list = list(map(float_list_to_str, features))
    return '\n'.join(features_string_list)


if __name__ == "__main__":
    input_txt = open("input.txt", "r").read()
    samples = get_total_input(input_txt)
    feature_string = generate_total_output(samples)
    
    with open("output.txt", "w") as f:
        f.write(feature_string)

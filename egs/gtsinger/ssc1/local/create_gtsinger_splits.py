import argparse
import os


def create_splits(wav_scp, train_set, dev_set, test_set):
    """Create train/dev/test splits from wav.scp file"""

    with open(wav_scp, "r") as f:
        lines = f.readlines()

    os.makedirs(os.path.dirname(train_set), exist_ok=True)
    os.makedirs(os.path.dirname(dev_set), exist_ok=True)
    os.makedirs(os.path.dirname(test_set), exist_ok=True)

    train_f = open(train_set, "w")
    dev_f = open(dev_set, "w")
    test_f = open(test_set, "w")

    # Note that these are just for the Serenade paper
    dev_keywords = ["Safe_and_Sound_"]
    test_keywords = ["What_Are_Words_"]

    # Sort lines into splits
    for line in lines:
        if "Speech_Group" in line:
            continue
        if any(kw in line for kw in dev_keywords):
            dev_f.write(line)
        elif any(kw in line for kw in test_keywords):
            if "Control_Group" in line:
                continue
            if "Vibrato_Group" in line:
                continue
            if "Glissando_Group" in line:
                continue
            test_f.write(line)
        else:
            train_f.write(line)

    # Close files
    train_f.close()
    dev_f.close()
    test_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wav-scp", type=str, required=True, help="Path to wav.scp file"
    )
    parser.add_argument(
        "--train-set", type=str, required=True, help="Path to output train set wav.scp"
    )
    parser.add_argument(
        "--dev-set", type=str, required=True, help="Path to output dev set wav.scp"
    )
    parser.add_argument(
        "--test-set", type=str, required=True, help="Path to output test set wav.scp"
    )
    args = parser.parse_args()

    create_splits(args.wav_scp, args.train_set, args.dev_set, args.test_set)

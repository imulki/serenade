import os
import glob
import argparse


def file_exists_in_any_subdirectory(base_path, file_name):
    # Get all directories matching the pattern
    matching_dirs = glob.glob(base_path)

    # Check each matching directory for the file
    for directory in matching_dirs:
        if os.path.exists(os.path.join(directory, file_name)):
            return True

    return False


def create_wav_scp(directory, output_file):
    """
    Create a Kaldi wav.scp file for all .flac files in the given directory.

    Args:
        directory (str): Path to the directory to search for .flac files.
        output_file (str): Name of the output wav.scp file (default is "wav.scp").
    """
    # Create base directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as wav_scp:
        # Get all wav files using glob
        wav_files = glob.glob(os.path.join(directory, "**/*.wav"), recursive=True)

        for file_path in wav_files:
            if "_reference" in file_path:
                continue

            # Get absolute path
            file_path = os.path.abspath(file_path)

            # Create utt_id by converting relative path to underscores
            relative_path = os.path.relpath(file_path, directory)
            utt_id = relative_path.replace(os.sep, "_").replace(".wav", "")
            utt_id = utt_id.replace("　", "_")
            utt_id = utt_id.replace(" ", "_")

            if "out." in utt_id:
                utt_id = "_".join(utt_id.split("_")[1:])

            wav_scp.write(f"{utt_id} {file_path}\n")
    print(f"wav.scp file has been created at {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    create_wav_scp(args.input_dir, args.output_file)

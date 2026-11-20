"""Download the Keio dataset via Kaggle CLI; dataset files are not committed."""
import argparse
import shutil
import subprocess
from pathlib import Path

DATASET = "romainpessia/artificial-lunar-rocky-landscape-dataset"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw/keio")
    args = parser.parse_args()
    if not shutil.which("kaggle"):
        raise SystemExit("Kaggle CLI missing. Install it with `pip install kaggle` and configure Kaggle API credentials.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", DATASET, "-p", str(output), "--unzip"], check=True)
    print(f"Downloaded files to {output}. Review the dataset card and license before use.")


if __name__ == "__main__": main()

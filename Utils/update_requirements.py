from pathlib import Path
import subprocess
from datetime import datetime
from Utils.install_dependencies import install_dependencies


def update_requirements_in():
    """Update requirements.in with all top-level packages except torch/torchvision."""
    in_file = "requirements.in"

    exclude = ["torch", "torchvision"]

    # Get top-level packages (like pip list --not-required)
    result = subprocess.run(
        ["pip", "list", "--not-required", "--format=freeze"],
        capture_output=True,
        text=True,
    )

    packages = {}
    for line in result.stdout.strip().split("\n"):
        if "==" in line:
            name, version = line.split("==")
            # if name.lower() not in exclude:
            packages[name] = version

    # Write to requirements.in
    with open(in_file, "a") as f:
        f.write(f"\n\n# ====== Updated: {datetime.now()} ====== \n")

        for name, version in sorted(packages.items()):
            if name in exclude:
                f.write(f"# {name}=={version}\n")
                continue

            f.write(f"{name}=={version}\n")

    print(f"{in_file} updated successfully with {len(packages)} packages.")


def dedupe_requirements_in(file_path="requirements.in"):
    path = Path(file_path)

    if not path.exists():
        print("No requirements.in file found.")
        return

    seen = set()
    new_lines = []

    with path.open("r") as f:
        for line in f:
            stripped = line.strip()

            if not stripped or stripped.startswith("#"):  # blank line  # comment
                new_lines.append(line)
                continue

            # Normalize requirement line (pip-tools ignores whitespace)
            normalized = stripped.lower()

            if normalized not in seen:
                seen.add(normalized)
                new_lines.append(line)
            else:
                print(f"Removing duplicate: {stripped}")

    # Write updated file
    with path.open("w") as f:
        f.writelines(new_lines)

    print("requirements.in cleaned and deduplicated.")


def restructure_requirements_in_file():
    file_name = "requirements.in"
    path = Path(file_name)
    now = datetime.now()

    # read file if exists
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            original_lines = f.readlines()
    else:
        original_lines = []

    # -----------------------------------------------------
    # 1. Remove any previous auto-header lines completely
    # -----------------------------------------------------
    def is_autogen(line: str):
        return (
            line.startswith("# torch==")
            or line.startswith("# torchvision==")
            or line.startswith("# ====== Updated:")
        )

    cleaned = [line for line in original_lines if not is_autogen(line)]

    # -----------------------------------------------------
    # 2. Deduplicate *package lines only*
    # -----------------------------------------------------
    seen = set()
    result = []
    for line in cleaned:
        stripped = line.strip()

        if stripped == "":
            result.append(line)
            continue

        if stripped.startswith("#"):
            result.append(line)
            continue

        normalized = stripped.lower()
        if normalized not in seen:
            seen.add(normalized)
            result.append(line)
        # else skip duplicates

    # -----------------------------------------------------
    # 3. Normalize blank lines (prevent growth)
    # -----------------------------------------------------
    normalized = []
    previous_blank = False

    for line in result:
        if line.strip() == "":
            # avoid multiple consecutive blanks
            if not previous_blank:
                normalized.append("\n")
            previous_blank = True
        else:
            normalized.append(line)
            previous_blank = False

    # Remove a blank line at start or end
    if normalized and normalized[0].strip() == "":
        normalized = normalized[1:]

    if normalized and normalized[-1].strip() == "":
        normalized = normalized[:-1]

    # -----------------------------------------------------
    # 4. Add clean header block at the top
    # -----------------------------------------------------
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    header = [
        f"# torch==2.9.0\n",
        f"# torchvision==0.24.0\n",
        f"# ====== Updated: {ts} ======\n",
        "\n",
    ]

    output = header + normalized + ["\n"]

    # -----------------------------------------------------
    # 5. Write final output
    # -----------------------------------------------------
    with path.open("w", encoding="utf-8") as f:
        f.writelines(output)

    print("requirements.in updated cleanly.")


def update_requirements_txt():
    """Update/Generate requirements.txt with all top-level packages except torch/torchvision."""
    subprocess.run(["pip", "install", "pip-tools"])

    # Get top-level packages (like pip list --not-required)
    subprocess.run(
        ["pip-compile", "requirements.in", "--output-file", "requirements.txt"]
    )

    print(f"Requirements.txt updated successfully.")


if __name__ == "__main__":
    # Make sure to install and sync dependencies before updating requirements
    install_dependencies()
    update_requirements_in()
    dedupe_requirements_in()
    restructure_requirements_in_file()
    update_requirements_txt()

import importlib.util
import subprocess
from pathlib import Path

from install_dependencies import install_dependencies


def update_requirements(output_file="requirements.txt"):
    """Update requirements.txt with all top-level packages except torch/torchvision."""
    exclude = ["torch", "torchvision"]

    # Get top-level packages (like pip list --not-required)
    result = subprocess.run(["pip", "list", "--not-required", "--format=freeze"], capture_output=True, text=True)

    packages = {}
    for line in result.stdout.strip().split("\n"):
        if "==" in line:
            name, version = line.split("==")
            if name.lower() not in exclude:
                packages[name] = version

    # Write to requirements.txt
    with open(output_file, "w") as f:
        for name, version in sorted(packages.items()):
            f.write(f"{name}=={version}\n")

    print(f"{output_file} updated successfully with {len(packages)} packages.")


if __name__ == "__main__":
    # Make sure to install and sync dependencies before updating requirements
    install_dependencies()
    update_requirements()

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

VENV_DIR = ".venv"


def _create_venv():
    """Create a virtual environment if it doesn't exist."""
    if not Path(VENV_DIR).exists():
        print(f"Creating virtual environment at {VENV_DIR}...")
        subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)
    else:
        print(f"Virtual environment {VENV_DIR} already exists.")


def _get_venv_python():
    """Return the path to the Python executable inside the venv."""
    if os.name == "nt":
        return str(Path(VENV_DIR) / "Scripts" / "python.exe")
    else:
        return str(Path(VENV_DIR) / "bin" / "python")


def _install_requirements(requirements_file="requirements.txt"):
    """Install all packages listed in requirements.txt using the venv Python."""
    venv_python = _get_venv_python()
    if not Path(requirements_file).exists():
        print(f"Error: {requirements_file} does not exist.")
        return
    print(f"Installing packages from {requirements_file} into {VENV_DIR}...")
    subprocess.run([venv_python, "-m", "pip", "install", "-r", requirements_file], check=True)
    print("All packages installed successfully!")


# ----------------- PyTorch Installation ----------------- #


def _is_installed(package_name):
    """Check if a package is installed."""
    spec = importlib.util.find_spec(package_name)
    return spec is not None


def _install_pytorch():
    if _is_installed("torch") and _is_installed("torchvision"):
        print("PyTorch and torchvision are already installed. Skipping installation.")
        return

    gpu_choice = (
        input(
            "Do you want to install pytorch and torchvision? Select install version:"
            "\n1. CUDA-compatible GPU"
            "\n2. CPU version"
            "\n3. Exit"
            "\n?: "
        )
        .strip()
        .lower()
    )

    if gpu_choice in ["1", "2"]:
        if gpu_choice == "1":
            # GPU installation - latest version with CUDA (default latest supported CUDA)
            cmd = [
                "pip3",
                "install",
                "torch",
                "torchvision",
                "--index-url",
                "https://download.pytorch.org/whl/cu126",
            ]
        else:
            # CPU-only installation
            cmd = ["pip3", "install", "torch", "torchvision"]

        print("Installing PyTorch...")
        subprocess.run(cmd, check=True)
        print("PyTorch installed successfully!")
        return

    print("Exiting without installing PyTorch.")


def install_dependencies():
    _create_venv()
    _install_requirements()
    _install_pytorch()


if __name__ == "__main__":
    install_dependencies()
    # print(Path("requirements.txt").resolve())
    # print(Path().cwd())
    # print(Path(VENV_DIR).resolve())

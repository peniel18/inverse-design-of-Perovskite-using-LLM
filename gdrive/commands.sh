

# setup.sh — Inverse Design of Perovskite using LLM

set -euo pipefail

echo "==============================================="
echo "Inverse Design of Perovskite using LLM"
echo "==============================================="

# Check required commands

check_command () {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Error: '$1' is not installed."
        exit 1
    }
}

check_command curl
check_command git
check_command python3


# Install uv

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -Ls https://astral.sh/uv/install.sh | sh

    # Add uv to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "uv already installed."
fi

# Clone repository

REPO_NAME="inverse-design-of-Perovskite-using-LLM"

if [ ! -d "$REPO_NAME" ]; then
    echo "Cloning repository..."
    git clone https://github.com/peniel18/inverse-design-of-Perovskite-using-LLM
else
    echo "Repository already exists."
fi

cd "$REPO_NAME"


# Create virtual environment

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
else
    echo "Virtual environment already exists."
fi


# Activate virtual environment

source .venv/bin/activate


# Install dependencies

echo "Installing dependencies..."
uv pip install -r requirements.txt


# Set PYTHONPATH

export PYTHONPATH="$(pwd)"

echo "PYTHONPATH set to:"
echo "$PYTHONPATH"


# check for .env file and load API_KEY

if [ -f ".env" ]; then
    export API_KEY=$(grep '^API_KEY=' .env | cut -d '=' -f2)

    if [ -z "$API_KEY" ]; then
        echo "Error: API_KEY not found in .env"
        exit 1
    fi

    echo "API_KEY loaded successfully."
else
    echo "Error: .env file not found."
    exit 1
fi



echo "Running data collection..."

python3 ./data/get_data_from_mp.py


# Validate Perovskites
echo "Validating Perovskite structures..."

python3 ./data/validate_perobskites.py


# Train/Test Split Placeholder

echo "Train/test split step not yet implemented."

# preprocess data
echo "Preprocessing data Pipeline..."

python3 ./data/cif_auditor.py --cif_dir ./cif_files --generate_template

python3 ./data/preprocessing_custom.py

python3 ./data/preprocessing_custom.py --cif_dir ./cif_files --label_csv label_template.csv

python3 ./data/fetch_labels.py


echo "==============================================="
echo "Setup completed successfully!"
echo "==============================================="
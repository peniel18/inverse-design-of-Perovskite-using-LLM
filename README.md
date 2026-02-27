# inverse-design-of-Perovskite-using-LLM
Leverage small language models for inverse design of Perovskite materials 


# Local Python Environment 
## 1. Install uv (if not already installed)

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

## 2. Clone the Repository 

```bash
git clone https://github.com/peniel18/inverse-design-of-Perovskite-using-LLM

cd  inverse-design-of-Perovskite-using-LLM
```


## Set up your virtual environment and activate 

```bash
uv venv 

source .venv/bin/activate  # linux/ mac os 
# or 
.venv\Scripts\activate # windows 

## Install dependencies 

uv pip install -r requirements.txt 
```


# Project collaboration guide 


## Creating a New Branch 
```bash 
git checkout -b feature/<your-feature-name>
```

## Making Commits 

```bash 

git add .
git commit -m "Comments on the commits"
```

## Pushing Changes 

```bash 
git push origin feature/<your-feature-name>
```


## Getting Data 
Before executing the get data script, make sure to set the PYTHONPATH environment variable so that Python can properly resolve the project modules.
```bash
export PYTHONPATH=$(pwd)

```


Now, we can execute the script 
```bash 
python3 ./data/get_data_from_mp.py

```


### 
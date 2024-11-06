

## Env setup

1. Mac
```bash
pyenv local 3.12.4
poetry env use $(pyenv which python)
poetry install
poetry shell
```

2. Linux

Run some sort of bootstrap first (TOOD: add it here?)

2.1 Python
```bash
add-apt-repository -y ppa:deadsnakes/ppa
apt install -y python3.12-full
python3.12 -m ensurepip
```

2.2 Poetry
```bash
curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.7.1 python3.12 -
  
echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Configure poetry settings
poetry config virtualenvs.in-project true
poetry config installer.parallel true

# Fix permissions
chown -R $SUDO_USER:$SUDO_USER $HOME/.local/share/pypoetry
chown -R $SUDO_USER:$SUDO_USER $HOME/.config/pypoetry
```

2.3 Install deps
```bash
poetry env use 3.12
poetry install
poetry shell
```

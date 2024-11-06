

## Env setup

```bash
pyenv local 3.12.4
poetry env use $(pyenv which python)
poetry install
poetry shell
```
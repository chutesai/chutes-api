FROM python:3.12
RUN useradd chutes -s /bin/bash -d /home/chutes && mkdir -p /home/chutes && chown chutes:chutes /home/chutes
ADD --chown=chutes . /app
USER chutes
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH=$PATH:/home/chutes/.local/bin
WORKDIR /app
RUN poetry install
ENTRYPOINT ["poetry", "run", "uvicorn", "run_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

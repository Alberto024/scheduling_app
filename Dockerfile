FROM python:3.8-slim-buster as python-base

EXPOSE 8501
WORKDIR /app/

# Install Build dependencies
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    # deps for installing poetry
    curl \
    # deps for building python deps
    build-essential

# Install Poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | POETRY_HOME=/opt/poetry python && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

# Copy poetry.lock* in case it doesn't exist in the repo
COPY pyproject.toml poetry.lock* /app/
RUN bash -c "poetry install --no-root --no-dev"

COPY ./app /app

CMD ["/bin/bash", "-c", "streamlit run --server.port $PORT /app/scheduling_app.py"]
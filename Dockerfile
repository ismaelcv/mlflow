FROM python:3.10.2

ARG MLFLOW_ENV

ENV YOUR_ENV=${MLFLOW_ENV} \
  PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=1.1.13


# System deps:
RUN pip install "poetry==$POETRY_VERSION"

# Copy only requirements to cache them in docker layer
WORKDIR /
COPY poetry.lock pyproject.toml /


# Project initialization:
RUN poetry config virtualenvs.create false \
  && poetry install $(test "$MLFLOW_ENV" == production && echo "--no-dev") --no-interaction --no-ansi

# Creating folders, and files for a project:
COPY . .

EXPOSE 8085

# Run your app
CMD [ "python", "./mlflow/apps/basic_app.py" ]

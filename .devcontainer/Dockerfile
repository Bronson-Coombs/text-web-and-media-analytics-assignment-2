# Base image
FROM python:3.12.2-slim

# Python Package Manager
RUN pip install poetry

# Copy build dependencies into image
COPY python/pyproject.toml .

# Copy version dependencies
COPY python/poetry.lock .

# Disable poetry creating virtual env in image and install dependencies
RUN poetry config virtualenvs.create false && poetry install --no-interaction
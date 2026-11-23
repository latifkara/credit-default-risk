FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir .
ENTRYPOINT ["customer-sat"]
CMD ["--help"]

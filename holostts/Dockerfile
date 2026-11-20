# Build context: the repository root.
ARG BUILD_FROM=ghcr.io/home-assistant/amd64-base-debian:trixie-2026.11.0

FROM ghcr.io/astral-sh/uv:0.13.5 AS uv

FROM $BUILD_FROM AS build
RUN apt-get update && apt-get install -y --no-install-recommends python3 git curl ca-certificates make \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /src
COPY pyproject.toml uv.lock README.md LICENSE Makefile ./
COPY src ./src
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON=/usr/bin/python3 UV_PYTHON_DOWNLOADS=never UV_COMPILE_BYTECODE=1
RUN uv sync --frozen --no-dev --no-editable
RUN make models MODEL_DIR=/models

FROM $BUILD_FROM
RUN apt-get update && apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/venv /opt/venv
COPY --from=build /models /models
COPY holostts/run.sh /run.sh
RUN chmod a+x /run.sh
ENV HOLOSTTS_MODELS=/models
LABEL org.opencontainers.image.source="https://github.com/nemethhh/holostts-wyoming" \
      org.opencontainers.image.description="HolosTTS Ukrainian text to speech for Home Assistant (Wyoming)" \
      org.opencontainers.image.licenses="MIT"
CMD ["/run.sh"]

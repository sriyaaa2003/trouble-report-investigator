FROM python:3.12-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 INVESTIGATOR_DATA_DIR=/data/mozilla \
    INVESTIGATOR_ARTIFACTS_DIR=/data/artifacts INVESTIGATOR_EMBEDDING_CACHE_DIR=/data/models
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .
VOLUME /data
EXPOSE 8000
# 1) docker run -v data:/data IMAGE investigator fetch --out /data/mozilla
# 2) docker run -v data:/data IMAGE investigator build
# 3) docker run -v data:/data -p 8000:8000 IMAGE
CMD ["investigator", "serve", "--host", "0.0.0.0", "--port", "8000"]

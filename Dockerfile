# CPU image — runs anywhere, no GPU or drivers needed. Good with --model tiny.
#   docker run -p 8010:8010 -v ~/.cache/huggingface:/root/.cache/huggingface \
#     ghcr.io/moudrkat/brainscope:cpu
FROM python:3.12-slim

# CPU-only torch first: it is the largest dependency and this keeps it in its
# own cached layer (and out of the CUDA wheel index).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . /src
RUN pip install --no-cache-dir /src && rm -rf /src

EXPOSE 8010
ENTRYPOINT ["brainscope", "--no-browser", "--host", "0.0.0.0"]
CMD ["--model", "tiny"]

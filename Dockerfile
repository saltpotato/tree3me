FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

WORKDIR /workspace/tree3

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

COPY tree_core.py .
COPY train_frontier_actorcritic.py .
COPY progress_server.py .
COPY README.md .

RUN mkdir -p models outputs

EXPOSE 80

CMD ["python", "train_frontier_actorcritic.py"]
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel

WORKDIR /workspace/tree3

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

COPY requirements-gpu.txt .
RUN pip install --no-cache-dir -r requirements-gpu.txt

COPY tree_core.py .
COPY train_policy_selfplay.py .
COPY progress_server.py .
COPY README.md .

RUN mkdir -p models outputs

EXPOSE 80

CMD ["python", "train_policy_selfplay.py"]
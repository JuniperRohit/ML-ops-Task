FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip && pip install -e .
EXPOSE 8030
CMD ["uvicorn", "agentic_mlops.api:app", "--host", "0.0.0.0", "--port", "8030"]

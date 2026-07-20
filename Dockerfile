FROM python:3.13.5-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

RUN groupadd -r app && useradd -r -m -g app app
RUN chown -R app:app /app

USER app

EXPOSE 8502

CMD ["streamlit", "run", "mask_detection.py", "--server.port=8502", "--server.address=0.0.0.0"]

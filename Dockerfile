FROM python:3.11-slim

WORKDIR /app

# 安装必要的编译环境给 curl_cffi
RUN apt-get update && apt-get install -y gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# PaaS 平台通常会覆盖这个环境变量，提供默认值 8080
ENV PORT=8080

CMD ["python", "main.py"]
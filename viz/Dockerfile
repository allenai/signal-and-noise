FROM python:3.9

WORKDIR /home

COPY ./requirements.txt /home/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /home/requirements.txt

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Create directories with proper permissions for pip installation
RUN mkdir -p /.local/lib/python3.9/site-packages && \
    chmod -R 777 /.local

# COPY ./data /data

COPY ./src /home

# Startup script
COPY ./startup.sh /home/startup.sh
RUN chmod +x /home/startup.sh

CMD ["/home/startup.sh"]

# CMD ["gunicorn", "main:server", "-b", "0.0.0.0:7860"]

FROM python:3.9

WORKDIR /home

COPY ./requirements.txt /home/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /home/requirements.txt

COPY ./data /data

COPY ./analysis /home

CMD ["gunicorn", "main:server", "-b", "0.0.0.0:7860"]
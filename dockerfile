FROM python:3.8.6-buster

COPY setup.py .setup.py
COPY requirements.txt /requirements.txt
COPY api /api
COPY CoolMelodyProject /CoolMelodyProject

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicon api.api:app --host 0.0.0.0 --port $PORT

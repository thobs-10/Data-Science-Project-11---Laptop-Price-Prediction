FROM python:3.10

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

RUN pip3 install evidently

COPY app.py .
COPY rf_reg.pkl .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=9696"]
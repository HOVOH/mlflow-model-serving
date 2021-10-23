FROM ghcr.io/hovoh/python-nltk:latest

RUN pip install numpy pandas boto3 scikit-learn modal mlflow flask waitress
WORKDIR /usr/src/app
COPY src .
COPY requirements.txt .
RUN pip install -r requirements.txt
EXPOSE 9000
CMD ["python", "main.py"]

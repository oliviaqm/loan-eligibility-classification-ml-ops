FROM python:3.7.4

WORKDIR /usr/src/loan-eligibility

COPY GBM_Model_version1.pkl ./GBM_Model_version1.pkl
COPY requirements.txt ./requirements.txt
COPY app.py ./app.py

RUN pip install -r requirements.txt

ENTRYPOINT ["python"]
CMD ["app.py"]


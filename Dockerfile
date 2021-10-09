FROM python:3.7.4

WORKDIR /usr/src/loan-eligibility

COPY GBM_Model_version1.pkl ./GBM_Model_version1.pkl
COPY LoansTrainingSetV2.csv ./LoansTrainingSetV2.csv
COPY test_data.csv ./test_data.csv
COPY Output_Test.csv ./Output_Test.csv
COPY requirements.txt ./requirements.txt
COPY app.py ./app.py

RUN pip install -r requirements.txt

ENTRYPOINT ["python"]
CMD ["app.py"]


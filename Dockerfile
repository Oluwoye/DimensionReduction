#syntax=docker/dockerfile:1

FROM python:latest

ENV DEBIAN_FRONTEND=noninteractive

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install tqdm

WORKDIR /
RUN mkdir results_direct
COPY . .
# COPY preprocessed_data/ /preprocessed_data/

ENV PYTHONPATH=/
RUN chmod +x code_ssnp/experiment1_direct.py
RUN ls -la .
CMD ["/usr/bin/env", "python3", "./code_ssnp/experiment1_direct.py", "--mode", "tfidf", "--n_jobs", "20", "--optimization", "random", "--random_permutations", "100"]
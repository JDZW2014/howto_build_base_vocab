FROM jdzw2014/python:3.8.5

ARG PIP=/usr/local/bin/pip

# for chatbot server
#RUNapscheduler $PIP install -i https://pypi.tuna.tsinghua.edu.cn/simple --default-timeout=100 sentencepiece
COPY requirements.txt requirements.txt
RUN $PIP install -r requirements.txt
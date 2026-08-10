FROM alpine:3.22

WORKDIR /app

RUN apk add --no-cache python3 py3-pip python3-dev build-base

#get some files we need
COPY run.sh ./
COPY requirements.txt ./

#install python 
RUN python3 -m venv .venv \
    && .venv/bin/pip install -r requirements.txt \


# They are angry about the name
# Why would they? how dare they tell me not to
COPY numpy_hello*.so ./numpy_hello.so

CMD ["sh", "run.sh"]
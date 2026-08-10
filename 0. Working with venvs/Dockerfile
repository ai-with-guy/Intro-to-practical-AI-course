FROM alpine:3.22

WORKDIR /app

RUN apk add --no-cache python3 py3-pip python3-dev build-base

#get some files we need
COPY run.sh ./
COPY requirements.txt ./
COPY numpy_20_hello.c ./
COPY build_numpy_20.py ./

#install python
RUN python3 -m venv .venv \
    && .venv/bin/pip install -r requirements.txt 

# we dont need to build it just copy it over
# RUN .venv/bin/python build_numpy_20.py


# They are angry about the name
# Why would they? how dare they tell me not to
COPY numpy_20_hello*.so ./numpy_20_hello.so

CMD ["sh", "run.sh"]

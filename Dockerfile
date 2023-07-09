FROM alpine:latest

RUN apk add build-base cmake py3-pip python3-dev \
    py3-pandas py3-matplotlib \
    libc-dev libstdc++-dev \
    bash bash-completion \
    &&  apk cache clean

RUN pip install gradio
RUN pip install mdtex2html
RUN pip install chatglm_cpp

COPY ./ /chatglm
WORKDIR /chatglm

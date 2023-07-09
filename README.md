```
docker build -t glm-chain .
docker run -it --rm -v $PWD/models:/chatglm/models glm-chain bash

python cli_demo.py -m models/chatglm2-ggml.bin -i
python web_demo.py -m models/chatglm2-ggml.bin
```

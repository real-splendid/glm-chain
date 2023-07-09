import argparse
from pathlib import Path

import chatglm_cpp

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "chatglm-ggml.bin"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=Path, help="model path")
    parser.add_argument("-p", "--prompt", default="Hello", type=str, help="prompt to start generation with")
    parser.add_argument("-i", "--interactive", action="store_true", help="run in interactive mode")
    parser.add_argument(
        "-l", "--max_length", default=2048, type=int, help="max total length including prompt and output"
    )
    parser.add_argument("-c", "--max_context_length", default=512, type=int, help="max context length")
    parser.add_argument("--top_k", default=0, type=int, help="top-k sampling")
    parser.add_argument("--top_p", default=0.7, type=float, help="top-p sampling")
    parser.add_argument("--temp", default=0.95, type=float, help="temperature")
    parser.add_argument("-t", "--threads", default=0, type=int, help="number of threads for inference")
    args = parser.parse_args()

    pipeline = chatglm_cpp.Pipeline(args.model)

    if not args.interactive:
        for piece in pipeline.stream_chat(
            [args.prompt],
            max_length=args.max_length,
            max_context_length=args.max_context_length,
            do_sample=args.temp > 0,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temp,
        ):
            print(piece, sep="", end="", flush=True)
        print()
        return

    history = []
    while True:
        try:
            prompt = input(f"{'Prompt':{len(pipeline.model.type_name)}} > ")
        except EOFError:
            break
        if not prompt:
            continue
        if prompt == "stop":
            break
        if prompt == "clear":
            history = []
            continue
        history.append(prompt)
        print(f"{pipeline.model.type_name} > ", sep="", end="")
        output = ""
        for piece in pipeline.stream_chat(
            history,
            max_length=args.max_length,
            max_context_length=args.max_context_length,
            do_sample=args.temp > 0,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temp,
        ):
            print(piece, sep="", end="", flush=True)
            output += piece
        print()
        history.append(output)
    print("Bye")


if __name__ == "__main__":
    main()

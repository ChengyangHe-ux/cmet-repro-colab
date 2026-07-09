import platform

import torch


def main() -> None:
    print("python:", platform.python_version())
    print("platform:", platform.platform())
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda_device:", torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print("total_vram_gb:", round(props.total_memory / 1024**3, 2))


if __name__ == "__main__":
    main()

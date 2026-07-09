import platform

try:
    import torch
except ModuleNotFoundError:
    torch = None


def main() -> None:
    print("Python 版本:", platform.python_version())
    print("系统平台:", platform.platform())
    if torch is None:
        print("Torch 状态: 未安装")
        print("CUDA 是否可用: 无法检查，因为当前 Python 环境没有安装 torch")
        return

    print("Torch 版本:", torch.__version__)
    print("CUDA 是否可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA 设备:", torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print("总显存 GB:", round(props.total_memory / 1024**3, 2))


if __name__ == "__main__":
    main()

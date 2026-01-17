#!/usr/bin/env python3
import os
import grp

import torch


def main() -> None:
    group_names = [grp.getgrgid(g).gr_name for g in os.getgroups()]
    print("groups:", group_names)

    path = "/dev/dri/renderD128"
    try:
        fd = os.open(path, os.O_RDWR)
        print("opened renderD128 OK")
        os.close(fd)
    except Exception as exc:
        print("open failed:", exc)

    print("torch", torch.__version__)
    print("xpu available", torch.xpu.is_available())
    if torch.xpu.is_available():
        print("xpu device", torch.xpu.get_device_name(0))


if __name__ == "__main__":
    main()

# # Check Pytorch installation
try:
    import timm
    import mmcv
    print('mmcv:', mmcv.__version__)  # 1.3.7<=x<=1.4.0, so install 1.4.0
    import torch
    import torchvision
    print(torch.__version__, torch.cuda.is_available())
    print(torchvision.__version__, torch.cuda.is_available())
except ModuleNotFoundError as e:
    print(e)


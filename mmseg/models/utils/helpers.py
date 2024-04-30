""" 
-*- coding: utf-8 -*-
    @Time    : 2023/3/3  10:31
    @Author  : AresDrw
    @File    : helpers.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""


def maybe_download(model_name, model_url, model_dir=None, map_location=None):
    import os
    import sys
    from six.moves import urllib

    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv("TORCH_HOME", "~/.torch"))
        model_dir = os.getenv("TORCH_MODEL_ZOO", os.path.join(torch_home, "models"))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = "{}.pth.tar".format(model_name)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        url = model_url
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urllib.request.urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)

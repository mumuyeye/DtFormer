from collections import OrderedDict
import torch


def publish_final_model(src, det, exp=None):
    state_dict = torch.load(src, map_location='cuda:0')
    model_dict = dict(meta=dict(), state_dict=OrderedDict())
    model_dict['meta'] = state_dict['meta']

    for key in state_dict['state_dict'].keys():
        if 'model' in key and 'teacher' not in key:
            model_dict['state_dict'][key] = state_dict['state_dict'][key]
    torch.save(model_dict, det)


def publish_teacher_model(src, det, exp=None):
    state_dict = torch.load(src, map_location='cuda:0')
    model_dict = dict(meta=dict(), state_dict=OrderedDict())

    for key in state_dict['state_dict'].keys():
        if exp in key and 'model' not in key:
            newkey = key.replace(exp, 'model')
            model_dict['state_dict'][newkey] = state_dict['state_dict'][key]
    torch.save(model_dict, det)


def change_key_name(src, det):
    state_dict = torch.load(src, map_location='cpu')
    model_dict = OrderedDict()

    for key in state_dict.keys():
        if 'image_encoder' in key:
            newkey = key.replace('image_encoder', 'backbone')
        model_dict[newkey] = state_dict[key]
    torch.save(model_dict, det)


if __name__ == "__main__":
    src_state_dict_path = '/hy-tmp/DAFormer/work_dir/iter_40000.pth'
    out_state_dict_path = '/hy-tmp/DAFormer/work_dir/segformer_model.pth'
    publish_final_model(src_state_dict_path, out_state_dict_path)

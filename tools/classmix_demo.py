import os
import os.path as osp
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
from mmseg.models.utils.visualization import colorize_mask, Cityscapes_palette


def loadImageFromFile(filename,
                      to_float32=False,
                      color_type='color',
                      backend='disk',
                      imdecode_backend='cv2',
                      size=None):
    file_client = mmcv.FileClient(backend=backend)
    img_bytes = file_client.get(filename)
    img = mmcv.imfrombytes(
        img_bytes, flag=color_type, backend=imdecode_backend)
    if to_float32:
        img = img.astype(np.float32)

    # ToTensor
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(dim=0)
    img = F.interpolate(img, size=size)
    return img


def loadAnnotations(filename,
                    reduce_zero_label=False,
                    backend='disk',
                    imdecode_backend='pillow',
                    size=None):
    file_client = mmcv.FileClient(backend=backend)
    img_bytes = file_client.get(filename)
    gt_semantic_seg = mmcv.imfrombytes(
        img_bytes, flag='unchanged',
        backend=imdecode_backend).squeeze().astype(np.uint8)
    # reduce zero_label
    if reduce_zero_label:
        # avoid using underflow conversion
        gt_semantic_seg[gt_semantic_seg == 0] = 255
        gt_semantic_seg = gt_semantic_seg - 1
        gt_semantic_seg[gt_semantic_seg == 254] = 255

    # ToTensor
    gt_semantic_seg = torch.from_numpy(gt_semantic_seg).unsqueeze(dim=0).unsqueeze(dim=0)
    gt_semantic_seg = F.interpolate(gt_semantic_seg, size=size)
    return gt_semantic_seg


def generate_class_mask(label, classes):
    label, classes = torch.broadcast_tensors(label,
                                             classes.unsqueeze(1).unsqueeze(2))
    class_mask = label.eq(classes).sum(0, keepdims=True)
    return class_mask


def get_class_masks(labels):
    class_masks = []
    for label in labels:
        classes = torch.unique(labels)
        nclasses = classes.shape[0]
        class_choice = np.random.choice(
            nclasses, int((nclasses + nclasses % 2) / 2), replace=False)
        classes = classes[torch.Tensor(class_choice).long()]
        class_masks.append(generate_class_mask(label, classes).unsqueeze(0))
    return class_masks


def mix(mask, data=None, target=None):
    if mask is None:
        return data, target
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0 * data[0] +
                (1 - stackedMask0) * data[1])
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0 * target[0] +
                  (1 - stackedMask0) * target[1])
    return data, target


def classmix_demo(src_img_path, src_label_path,
                  dst_img_path, pseudo_label, out_dir):
    src_img = loadImageFromFile(src_img_path, size=(512, 1024))
    dst_img = loadImageFromFile(dst_img_path, size=(512, 1024))
    gt_semantic_seg = loadAnnotations(src_label_path, size=(512, 1024))
    pseudo_label = loadAnnotations(pseudo_label, size=(512, 1024))

    os.makedirs(out_dir, exist_ok=True)

    mix_masks = get_class_masks(gt_semantic_seg)

    mixed_img, mixed_lbl = mix(
        mix_masks,
        data=torch.stack((src_img[0], dst_img[0])),
        target=torch.stack((gt_semantic_seg[0], pseudo_label[0])))

    # save the domain_mask:
    plt.imsave(fname=osp.join(out_dir, 'domain_mask.png'),
               arr=mix_masks[0][0][0].numpy(),
               cmap='gray')

    # save the mixed_image
    plt.imsave(fname=osp.join(out_dir, 'mixed_image.png'),
               arr=mixed_img[0].permute(1, 2, 0).cpu().detach().numpy().astype('uint8'))

    # save the mixed_label:
    color_img = colorize_mask(mixed_lbl[0][0].cpu().numpy(), palette=Cityscapes_palette)  # 'PIL'
    color_img.save(osp.join(out_dir, f'mixed_label_color.png'))

    print('done')


if __name__ == "__main__":
    src_image = '/raid/wzq/datasets/paper/xformer_cum/s2sf_mix/aachen_000004_000019_leftImg8bit.png'
    src_label = '/raid/wzq/datasets/paper/xformer_cum/aachen_000004_000019_gtFine_labelTrainIds.png'
    dst_image = '/raid/wzq/datasets/paper/xformer_cum/s2sf_mix/syn.png'
    dst_label = '/raid/wzq/datasets/paper/xformer_cum/s2sf_mix/syn_gt_labelTrainIds.png'
    classmix_demo(src_image, src_label, dst_image, dst_label, out_dir='/raid/wzq/datasets/paper/xformer_cum/s2sf_mix/')

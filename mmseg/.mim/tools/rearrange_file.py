import os
import shutil
if __name__ == "__main__":
    img_dir = '/raid/wzq/data/DA_airbone_satellite/satellite/image'
    copy_img_dir = '/raid/wzq/data/DA_airbone_satellite/satellite/image2'
    ann_dir = '/raid/wzq/data/DA_airbone_satellite/satellite/label'
    copy_ann_dir = '/raid/wzq/data/DA_airbone_satellite/satellite/label2'
    for i, file in enumerate(os.listdir(img_dir)):
        if file in os.listdir(ann_dir):
            shutil.copy(os.path.join(img_dir, file),
                        os.path.join(copy_img_dir, f'{i}.tif'))
            shutil.copy(os.path.join(ann_dir, file),
                        os.path.join(copy_ann_dir, f'{i}.tif'))
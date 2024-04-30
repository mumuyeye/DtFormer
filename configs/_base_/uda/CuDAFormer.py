# Baseline UDA
uda = dict(
    type='CuDAFormer',
    alpha=0.99,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    pseudo_ref_weight_ignore_top=150,
    use_ref_label=True,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    debug_img_interval=1000,
)

use_ddp_wrapper = True

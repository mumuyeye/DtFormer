# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# Baseline UDA
uda = dict(
    type='NoUDA',
    debug_img_interval=1000,
    print_grad_magnitude=False
)
use_ddp_wrapper = True

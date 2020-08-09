# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

from . import submission, util
from .submission.run_context import RunContext
from .submission.submit import (PathType, SubmitConfig, SubmitTarget,
                                convert_path, get_path_from_template,
                                make_run_dir_path, submit_run)
from .util import EasyDict

submit_config: SubmitConfig = None  # Package level variable for SubmitConfig which is only valid when inside the run function.

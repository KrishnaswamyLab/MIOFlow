__all__ = ['ROOT_DIR', 'DATA_DIR', 'NTBK_DIR', 'IMGS_DIR', 'RES_DIR', 'WORM_FILE', 'EB_BODIES_FILE', 'EB_BODIES_PSEUDO_4',
           'EB_BODIES_PSEUDO_6', 'EB_BODIES_PSEUDO_25', 'EB_BODIES_PSEUDO_82', 'DYNGEN_INFO_FILE', 'DYNGEN_EXPR_FILE']


'''
---
description: Includes `ROOT_DIR`, `DATA_DIR`, `NTBK_DIR`, and `IMGS_DIR`. `ROOT_DIR`
  is the location of this repository. The rest are scoped underneath and are default
  locations for data, notebooks, and images respectively.
output-file: constants.html
title: Constants

---
'''

import os, inspect
__file = inspect.getfile(lambda: None)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file), '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
NTBK_DIR = os.path.join(ROOT_DIR, 'notebooks')
IMGS_DIR = os.path.join(ROOT_DIR, 'imgs')
RES_DIR = os.path.join(ROOT_DIR, 'results')
WORM_FILE = os.path.join(DATA_DIR, 'worm_TrNet2.npz')
EB_BODIES_FILE = os.path.join(DATA_DIR, 'natalia_eb_rna_smoothed.npz')
EB_BODIES_PSEUDO_4 = os.path.join(DATA_DIR, 'pseudotime-4x.npy')
EB_BODIES_PSEUDO_6 = os.path.join(DATA_DIR, 'pseudotime-6x.npy')
EB_BODIES_PSEUDO_25 = os.path.join(DATA_DIR, 'pseudotime-25x.npy')
EB_BODIES_PSEUDO_82 = os.path.join(DATA_DIR, 'pseudotime-82x.npy')

DYNGEN_INFO_FILE = os.path.join(DATA_DIR, 'cell_info.csv')
DYNGEN_EXPR_FILE = os.path.join(DATA_DIR, 'dyngen_expression_bif.csv')
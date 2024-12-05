""" IEEE TCDS 2024 の比較実験に使うためのモジュール """

import pandas as pd

# TRAIN_DATA = [
#     [0, 0, 0, 0],
#     [1, 2, 1, 2],
#     [0, 0, 0, 0],
#     [1, 2, 1, 2],
# ]
TRAIN_DATA = pd.read_csv("../datas/train_zs_30.csv").values.tolist()
TRAIN_DATA.extend(pd.read_csv("../datas/train_extended_zs_70.csv").values.tolist())


# TEST_DATA = [
#     [0, 1, 0, 1],
#     [1, 3, 1, 3],
#     [0, 0, 0, 0],
#     [1, 2, 1, 2],
#     [0, 0, 0, 0],
#     [1, 2, 1, 2],
# ]
TEST_DATA = pd.read_csv("../datas/test_zs_10.csv").values.tolist()

assert len(TRAIN_DATA) == 100 ## 仮コード
assert len(TEST_DATA) == 10 ## 仮コード

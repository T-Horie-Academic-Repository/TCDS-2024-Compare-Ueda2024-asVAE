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

assert len(TRAIN_DATA) == 100 ## 仮コード

def get_test_data(num_char_sorts: int, exp_id: int, pred_id: int) -> list[list[int]]: 
    """
    The code to get the test data for the comparison experiment of IEEE TCDS 2024.
    """
    path = (
        f"../datas/Ueda{num_char_sorts:02}_prediction/predict_result_summary_variables"
        f"_Ueda{num_char_sorts:02}_exp{exp_id:03}_predict{pred_id:03}_attended_true_zs.csv"
    )
    return pd.read_csv(path).values.tolist()
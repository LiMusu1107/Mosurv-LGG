import os
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from util.models import init_model_dict, init_optim, load_model
from util.utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor, gen_test_adj_mat_tensor, cal_adj_mat_parameter
from util.utils import save_model_dict, load_model_dict,load_data
from util.train_test import prepare_trte_data, gen_trte_adj_mat,train_epoch,test_epoch




train_folder = "./train"
test_folder = "./test"
view_list = [1, 2, 3]

data_train_list, data_all_list, idx_dict = load_data(train_folder, test_folder, view_list)

adj_tr_list, adj_te_list = gen_trte_adj_mat(data_train_list, data_all_list, idx_dict, adj_parameter = 3)


num_view = len(view_list)

model_path = "./models"

model = load_model(num_view, model_path)

te_prob = test_epoch(data_all_list, adj_te_list, idx_dict["te"], model)

for i, probs in enumerate(te_prob, start=1):
    high_risk_prob = probs[0]
    low_risk_prob = probs[1]

    if low_risk_prob > high_risk_prob:
        print(f"The patient{i} is in the low-risk prognosis group, with a probability of {low_risk_prob * 100:.2f}%.")
    else:
        print(f"The patient{i} is in the high-risk prognosis group, with a probability of {high_risk_prob * 100:.2f}%.")
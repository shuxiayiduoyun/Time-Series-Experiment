#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2025/4/6 20:27
 @Author  : wly
 @File    : exp_long_term_forecasting.py.py
 @Description: 
"""
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def __build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        return model

    def __get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

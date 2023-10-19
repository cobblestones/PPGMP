#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from pprint import pprint
from utils import log
import sys
import time

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================
        self.parser.add_argument('--exp', type=str, default='test', help='ID of experiment')
        self.parser.add_argument('--is_eval', dest='is_eval', action='store_true',
                                 help='whether it is to evaluate the model')
        self.parser.add_argument('--ckpt', type=str, default='checkpoint/', help='path to save checkpoint')
        self.parser.add_argument('--skip_rate', type=int, default=5, help='skip rate of samples')
        self.parser.add_argument('--skip_rate_test', type=int, default=5, help='skip rate of samples for test')

        # ===============================================================
        #                     Model options
        # ===============================================================
        self.parser.add_argument('--in_features', type=int, default=54, help='size of each model layer')
        self.parser.add_argument('--num_stage', type=int, default=2, help='size of each model layer')
        self.parser.add_argument('--d_model', type=int, default=256, help='past frame number')
        self.parser.add_argument('--kernel_size', type=int, default=5, help='past frame number')
        self.parser.add_argument('--drop_out', type=float, default=0.3, help='drop out probability')
        self.parser.add_argument('--fine_tuning_number', type=int, default=4, help='the model number in the fine-tune stage')
        self.parser.add_argument('--l1norm', type=float, default=0,help='the weight of L1 norm')
        self.parser.add_argument('--GR_in_SFEN',dest='Global_Res_in_SFEN',  action='store_false',
                                 help='determine whether to use global residual in spatial feature extract network')
        self.parser.add_argument('--LR_in_SFEN',dest='local_Res_in_SFEN', action='store_false',
                                 help='determine whether to use local residual in spatial feature extract network')
        self.parser.add_argument('--GR_in_TFEN',dest='Global_Res_in_TFEN', action='store_false',
                                 help='determine whether to use global residual in temporal feature extract network')
        self.parser.add_argument('--LR__in_TFEN', dest='local_Res_in_TFEN', action='store_false',
                                 help='determine whether to use local residual in temporal feature extract network')

        # ===============================================================
        #                     Running options
        # ===============================================================
        self.parser.add_argument('--input_n', type=int, default=50, help='past frame number')
        self.parser.add_argument('--output_n', type=int, default=10, help='future frame number')
        self.parser.add_argument('--test_output_n', type=int, default=25, help='future frame number')
        self.parser.add_argument('--dct_n', type=int, default=10, help='future frame number')
        self.parser.add_argument('--lr_now', type=float, default=0.0005)
        self.parser.add_argument('--max_norm', type=float, default=10000)
        self.parser.add_argument('--epoch', type=int, default=100)
        self.parser.add_argument('--batch_size', type=int, default=32)
        self.parser.add_argument('--test_batch_size', type=int, default=128)
        self.parser.add_argument('--is_load', dest='is_load', action='store_true',
                                 help='whether to load existing model')

    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()

        self.current_train_id = time.strftime("%Y_%m_%d_%HH_%MM_%SS", time.localtime())

        if not self.opt.is_eval:
            script_name = os.path.basename(sys.argv[0])[:-3]
            log_name = '{}_bs{}_in{}_out{}_numstage{}_ks{}_dctn{}_L1norm{}_{}'.format(script_name, self.opt.batch_size,
                                                          self.opt.input_n,
                                                          self.opt.output_n,
                                                          self.opt.num_stage,
                                                          self.opt.kernel_size,
                                                          self.opt.dct_n,
                                                          self.opt.l1norm,
                                                          self.current_train_id)
            self.opt.exp = log_name
            # do some pre-check
            ckpt = os.path.join(self.opt.ckpt, self.opt.exp)
            if not os.path.isdir(ckpt):
                os.makedirs(ckpt)
                log.save_options(self.opt)
            self.opt.ckpt = ckpt
            log.save_options(self.opt)
        self._print()
        # log.save_options(self.opt)
        return self.opt

from utils import dpw3_3d as PW3_Motion3D
from model import Model
from utils.opt import Options
from utils import util
from utils import log
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
import torch.optim as optim
from tqdm import tqdm


def main(opt):
    lr_now = opt.lr_now
    start_epoch = 1
    print('>>> create models')
    in_features = opt.in_features
    d_model = opt.d_model
    kernel_size = opt.kernel_size
    net_pred = Model.AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                 num_stage=opt.num_stage, dct_n=opt.dct_n)
    net_pred.cuda()
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.lr_now)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))

    if opt.is_load or opt.is_eval:
        if opt.is_eval:
            model_path_len = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
        else:
            model_path_len = './{}/ckpt_last.pth.tar'.format(opt.ckpt)
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        ckpt = torch.load(model_path_len)
        start_epoch = ckpt['epoch'] + 1
        lr_now = ckpt['lr']
        net_pred.load_state_dict(ckpt['state_dict'])
        print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

    print('>>> loading datasets')

    if not opt.is_eval:

        dataset = PW3_Motion3D.Datasets(opt, split=0)
        print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
        data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        valid_dataset = PW3_Motion3D.Datasets(opt, split=1)
        print('>>> Validation dataset length: {:d}'.format(valid_dataset.__len__()))
        valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=True, num_workers=0,
                                  pin_memory=True)

    test_dataset = PW3_Motion3D.Datasets(opt, split=2)
    print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                             pin_memory=True)

    dim_used = dataset.dim_used

    # evaluation
    if opt.is_eval:
        ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt, dim_used=dim_used)
        ret_log = np.array([])
        head = np.array([])
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
            head = np.append(head, [k])
        log.save_csv_log(opt, head, ret_log, is_create=True, file_name='test_walking')

    # training
    if not opt.is_eval:
        err_best = 1000
        for epo in range(start_epoch, opt.epoch + 1):
            is_best = False
            # if epo % opt.lr_decay == 0:
            lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / opt.epoch))

            print('>>> training epoch: {:d}'.format(epo))
            ret_train = run_model(net_pred, optimizer, is_train=0, data_loader=data_loader, epo=epo, opt=opt, dim_used=dim_used)
            print('train error: {:.3f}'.format(ret_train['m_p3d_h36']))

            ret_valid = run_model(net_pred, is_train=1, data_loader=valid_loader, opt=opt, epo=epo, dim_used=dim_used)
            print('validation error: {:.3f}'.format(ret_valid['m_p3d_h36']))

            ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt, epo=epo, dim_used=dim_used)
            print('testing error: {:.3f}'.format(ret_test['#40ms']))

            ret_log = np.array([epo, lr_now])
            head = np.array(['epoch', 'lr'])
            for k in ret_train.keys():
                ret_log = np.append(ret_log, [ret_train[k]])
                head = np.append(head, [k])
            for k in ret_valid.keys():
                ret_log = np.append(ret_log, [ret_valid[k]])
                head = np.append(head, ['valid_' + k])
            for k in ret_test.keys():
                ret_log = np.append(ret_log, [ret_test[k]])
                head = np.append(head, ['test_' + k])
            log.save_csv_log(opt, head, ret_log, is_create=(epo == 1))
            valid_value = ret_valid['m_p3d_h36']
            if valid_value < err_best:
                err_best = valid_value
                is_best = True
            average_error = np.mean(
                (ret_test["#200ms"], ret_test["#400ms"], ret_test["#600ms"], ret_test["#800ms"], ret_test["#1000ms"]))
            err_value = 'AverageError{:.4f}_err{:.4f}_err{:.4f}_err{:.4f}_err{:.4f}_err{:.4f}'.format(
                average_error, ret_test["#200ms"], ret_test["#400ms"], ret_test["#600ms"], ret_test["#800ms"],ret_test["#1000ms"])
            log.save_ckpt(epo, lr_now, err_value,
                          {'epoch': epo, 'lr': lr_now,
                           'err': ret_valid['m_p3d_h36'],
                           'state_dict': net_pred.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          is_best=is_best, opt=opt)


def eval(opt):
    in_features = opt.in_features
    d_model = opt.d_model
    kernel_size = opt.kernel_size
    print('>>> create models')
    net_pred = Model.AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                 num_stage=opt.num_stage, dct_n=opt.dct_n)
    net_pred.to(opt.cuda_idx)
    net_pred.eval()

    # load model
    model_path_len = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
    print(">>> loading ckpt len from '{}'".format(model_path_len))
    ckpt = torch.load(model_path_len)
    net_pred.load_state_dict(ckpt['state_dict'])

    print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

    dataset = PW3_Motion3D.Datasets(opt=opt, split=2)
    dim_used = dataset.dim_used
    data_loader = DataLoader(dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                                  pin_memory=True)

    ret_test = run_model(net_pred, is_train=3, data_loader=data_loader, opt=opt, dim_used=dim_used)
    ret_log = np.array(['avg'])
    head = np.array(['action'])

    for k in ret_test.keys():
        ret_log = np.append(ret_log, [ret_test[k]])
        head = np.append(head, ['test_' + k])

    log.save_csv_eval_log(opt, head, ret_log, is_create=True)


def smooth(src, sample_len, kernel_size):
    """
    data:[bs, 60, 96]
    """
    src_data = src[:, -sample_len:, :].clone()
    smooth_data = src_data.clone()
    for i in range(kernel_size, sample_len):
        smooth_data[:, i] = torch.mean(src_data[:, kernel_size:i+1], dim=1)
    return smooth_data

def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None, dim_used=None):
    if is_train == 0:
        net_pred.train()
    else:
        net_pred.eval()

    l_p3d = 0
    n = 0
    in_n = opt.input_n
    if is_train <= 1:
        m_p3d_h36 = 0
        itera = 1
        out_n = opt.output_n
    else:
        itera = 3
        out_n = opt.test_output_n
        titles = (np.array(range(out_n)) + 1)*40
        m_p3d_h36 = np.zeros([out_n])

    seq_in = opt.kernel_size

    st = time.time()
    for i, (p3d_h36) in tqdm(enumerate(data_loader)):
        batch_size, seq_n, all_dim = p3d_h36.shape
        # when only one sample in this batch
        if batch_size == 1 and is_train == 0:
            continue
        n += batch_size
        bt = time.time()

        p3d_h36 = p3d_h36.float().cuda()

        input = p3d_h36[:, :, dim_used].clone()

        p3d_sup = p3d_h36.clone()[:, :, dim_used][:, -out_n - seq_in:].reshape(
            [-1, seq_in + out_n, len(dim_used) // 3, 3])

        p3d_out_all_0, p3d_out_all_1, p3d_out_all,weight = net_pred(input, input_n=in_n, output_n=10, itera=itera)

        p3d_out = p3d_h36.clone()[:, in_n:in_n + out_n]
        if is_train  <= 1:
            p3d_out[:, :, dim_used] = p3d_out_all[:, seq_in:, 0, ]
        elif itera==1:
            p3d_out[:, :, dim_used] = p3d_out_all[:, seq_in:, 0, ]
        else:
            p3d_out[:, :, dim_used] = p3d_out_all[:, seq_in:].transpose(1, 2).reshape([batch_size, 10 * itera, -1])[:,
                                      :out_n]

        p3d_out = p3d_out.reshape([-1, out_n, all_dim//3, 3])
        p3d_h36 = p3d_h36.reshape([-1, in_n + out_n, all_dim//3, 3])

        # 2d joint loss:
        grad_norm = 0
        if is_train == 0:

            p3d_out_all = p3d_out_all.reshape([batch_size, seq_in + out_n, len(dim_used) // 3, 3])
            p3d_out_all_1 = p3d_out_all_1.reshape([batch_size, seq_in + out_n, len(dim_used) // 3, 3])
            p3d_out_all_0 = p3d_out_all_0.reshape([batch_size, seq_in + out_n, len(dim_used) // 3, 3])

            loss_p3d_3 = torch.mean(torch.norm(p3d_out_all - p3d_sup, dim=3))
            loss_p3d_2 = torch.mean(torch.norm(p3d_out_all_1 - p3d_sup, dim=3))
            loss_p3d_1 = torch.mean(torch.norm(p3d_out_all_0 - p3d_sup, dim=3))

            loss_all = (loss_p3d_3 + loss_p3d_2 + loss_p3d_1)/3
            loss_l1 = torch.mean(torch.norm(weight, p=1, dim=0))
            loss_all = loss_all + opt.l1norm * loss_l1

            optimizer.zero_grad()
            loss_all.backward()
            nn.utils.clip_grad_norm_(list(net_pred.parameters()), max_norm=opt.max_norm)
            optimizer.step()
            l_p3d += loss_p3d_3.cpu().data.numpy() * batch_size

        if is_train <= 1:
            mpjpe_p3d_h36 = torch.mean(torch.norm(p3d_h36[:, in_n:in_n + out_n] - p3d_out, dim=3))
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy() * batch_size
        else:
            mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out, dim=3), dim=2), dim=0)
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()
        if i % 1000 == 0:
            print('{}/{}|bt {:.3f}s|tt{:.0f}s|gn{}'.format(i + 1, len(data_loader), time.time() - bt,
                                                           time.time() - st, grad_norm))

    ret = {}
    if is_train == 0:
        ret["l_p3d"] = l_p3d / n

    if is_train <= 1:
        ret["m_p3d_h36"] = m_p3d_h36 / n
    else:
        m_p3d_h36 = m_p3d_h36 / n
        for j in range(out_n):
            ret["#{:d}ms".format(titles[j])] = m_p3d_h36[j]
    return ret

if __name__ == '__main__':

    option = Options().parse()
    if option.is_eval == False:
        main(option)
    else:
        eval(option)
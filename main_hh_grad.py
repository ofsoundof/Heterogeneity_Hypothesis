
import torch
from util import utility
import os
from data import Data
from model_hh import Model
from loss import Loss
from util.trainer_hh_grad import Trainer
from util.option_dhp import args
from tensorboardX import SummaryWriter
from model_dhp.flops_counter_dhp import set_output_dimension, get_parameters, get_flops
# from IPython import embed

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)


def binary_search(model, target):
    """
    Binary search algorithm to determine the threshold
    :param model:
    :param target:
    :param merge_flag:
    :return:
    """
    # target = 0.70
    # threshold = model.get_model().args.threshold
    step = 0.01
    # step_min = 0.0001
    status = 1.0
    stop = 0.001
    counter = 1
    max_iter = 100
    flops = get_flops(model)
    params = get_parameters(model)

    while abs(status - target) > stop and counter <= max_iter:
        status_old = status
        # calculate flops and status
        model.set_parameters()
        flops_prune = get_flops(model)
        status = flops_prune / flops
        params_prune = get_parameters(model)
        params_compression_ratio = params_prune / params

        string = 'Iter {:<3}: current step={:1.8f}, current threshold={:2.8f}, status (FLOPs ratio) = {:2.4f}, ' \
                 'params ratio = {:2.4f}.\n'\
            .format(counter, step, model.pt, status, params_compression_ratio)
        print(string)

        if abs(status - target) > stop:
            # calculate the next step
            flag = False if counter == 1 else (status_old >= target) == (status < target)
            if flag:
                step /= 2
            # calculate the next threshold
            if status > target:
                model.pt += step
            elif status < target:
                model.pt -= step
                model.pt = max(model.pt, 0)

            counter += 1
            # deal with the unexpected status
            if model.pt < 0 or status <= 0:
                print('Status {} or threshold {} is out of range'.format(status, model.pt))
                break
        else:
            print('The target compression ratio is achieved. The loop is stopped')


if checkpoint.ok:
    """
    Four phases.
    Phase 1: training from scratch.                   load = '', pretrain = '' 
             -> not loading the model
             -> not loading the optimizer
    Phase 2: testing phase; test_only.                load = '', pretrain = '*/model/model_latest.pt' or '*/model/model_merge_latest.pt'
             -> loading the pretrained model
             -> not loading the optimizer
    Phase 3: loading models for PG optimization.      load = '*/' -- a directory, pretrain = '', epoch_continue = None
             -> loading from model_latest.pt
             -> loading optimizer.pt
    Phase 4: loading models to continue the training. load = '*/' -- a directory, pretrain = '', epoch_continue = a number
             -> loading from model_continue.pt
             -> loading optimizer_converging.pt
    During the loading phase (3 & 4), args.load is set to a directory. The loaded model is determined by the 'stage' of 
            the algorithm. 
    Thus, need to first determine the stage of the algorithm. 
    Then decide whether to load model_latest.pt or model_continue_latest.pt

    The algorithm has two stages, i.e, proximal gradient (PG) optimization (searching stage) and continue-training 
            (converging stage). 
    The stage is determined by epoch_continue. The variable epoch_continue denotes the epoch where the PG optimization
            finishes and the training continues until the convergence of the network.
    i.  epoch_continue = None -> PG optimzation stage (searching stage)
    ii. epoch_continue = a number -> continue-training stage (converging stage)

    Initial epoch_continue:
        Phase 1, 2, &3 -> epoch_continue = None, converging = False
        PHase 4 -> epoch_continue = a number, converging = True
    """

    # ==================================================================================================================
    # Step 1: Initialize the objects.
    # ==================================================================================================================
    info_path = os.path.join(checkpoint.dir, 'epochs_converging.pt')
    if args.load != '' and os.path.exists(info_path):
        # converging stage
        epoch_continue = torch.load(info_path)
    else:
        # searching stage
        epoch_continue = None
    # Judge which stage the algorithm is in, namely the searching stage or the converging stage.
    converging = False if epoch_continue is None else True

    loss = Loss(args, checkpoint)
    network_model = Model(args, checkpoint, converging=converging)
    network_model_teacher = Model(args, checkpoint, teacher=True) if args.distillation_final else None
    writer = SummaryWriter(checkpoint.dir, comment='searching') if args.summary else None

    loader = Data(args)
    t = Trainer(args, loader, network_model, loss, checkpoint, writer, converging, network_model_teacher)

    # ==================================================================================================================
    # Step 2: Searching -> use proximal gradient method to optimize the hypernetwork parameter and
    #          search the potential backbone network configuration
    #          Model: a sparse model
    # ==================================================================================================================
    if not converging and not args.test_only:
        # In the training phase or loading for searching
        # t.model = t.model_teacher
        # t.test() # test whether the teacher model is correctly loaded.
        while not t.terminate():
            t.train()
            t.test()

        binary_search(t.model.get_model(), args.ratio)
        if args.summary:
            t.writer.close()
        # save the compression ratio log and per-layer compression ratio for the latter use.
        t.model.get_model().per_layer_compression_ratio(0, 0, checkpoint.dir, save_pt=True)
        t.model.get_model().per_layer_compression_ratio(0, 0, checkpoint.dir)

    # ==================================================================================================================
    # Step 3: Pruning -> prune the derived sparse model and prepare the trainer instance for finetuning or testing
    # ==================================================================================================================

    t.reset_after_searching()
    if args.print_model:
        print(t.model.get_model())
        print(t.model.get_model(), file=checkpoint.log_file)

    # ==================================================================================================================
    # Step 4: Fintuning/Testing -> finetune the pruned model to have a higher accuracy.
    # ==================================================================================================================

    while not t.terminate():
        t.train()
        t.test()

    set_output_dimension(network_model.get_model(), t.input_dim)
    flops = get_flops(network_model.get_model())
    params = get_parameters(network_model.get_model())
    print('\nThe computation complexity and number of parameters of the current network is as follows.'
          '\nFlops: {:.4f} [G]\tParams {:.2f} [k]\n'.format(flops / 10. ** 9, params / 10. ** 3))

    if args.summary:
        t.writer.close()
    if args.print_model:
        print(t.model.get_model())
        print(t.model.get_model(), file=checkpoint.log_file)
    checkpoint.done()




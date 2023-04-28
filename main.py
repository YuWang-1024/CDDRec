# -*- coding: utf-8 -*-

import os
import numpy as np
import random
import torch
import argparse
import time

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import CDDRecDataset
from trainers import  CDDRecTrainer
from models import  CDDRecModel
from utils import EarlyStopping, get_user_seqs,  check_path, set_seed

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--data_name', default='Office_Products', type=str)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--ckp', default=10, type=int, help="pretrain epochs 10, 20, 30...")

    # model args
    parser.add_argument("--experimentation", default='', type=str, help="additional token for different training expeirments for the same model")
    parser.add_argument("--model_name", default='CDDRec', type=str)
    parser.add_argument("--hidden_size", type=int, default=128, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=1, help="number of layers")
    parser.add_argument('--num_attention_heads', default=4, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str) # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.2, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.0, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=20, type=int)
    parser.add_argument('--T', default=20, type=int)
    parser.add_argument('--beta_1', default=1e-4, type=float)
    parser.add_argument('--beta_T', default=0.002, type=float)


    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--data_augmentation', action="store_true")
    parser.add_argument('--linear_infonce', action="store_true")
    parser.add_argument('--loss_type', type = str, default = 'BPR', help = 'BPR or CE')

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=128, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    

    parser.add_argument("--load_model", action="store_true")




    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    args.data_file = args.data_dir + args.data_name + '.txt'
    user_seq, max_item, valid_rating_matrix, test_rating_matrix, num_users = \
        get_user_seqs(args.data_file)

    args.item_size = max_item + 2
    args.num_users = num_users
    args.mask_id = max_item + 1

    # save model args
    args_str = f'{args.experimentation}_{args.model_name}-{args.data_name}-{args.hidden_size}-{args.num_hidden_layers}-{args.num_attention_heads}-{args.hidden_act}-{args.attention_probs_dropout_prob}-{args.hidden_dropout_prob}-{args.max_seq_length}-{args.lr}-{args.weight_decay}-{args.ckp}-{args.T}-{args.beta_1}-{args.beta_T}-{args.linear_infonce}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    print(str(args))
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix


    # save model
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

   


    if args.model_name == 'CDDRec': 
        train_dataset = CDDRecDataset(args, user_seq, data_type='train')
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

        eval_dataset = CDDRecDataset(args, user_seq, data_type='valid')
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)


        test_dataset = CDDRecDataset(args, user_seq, data_type='test')
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)
        model = CDDRecModel(args=args)
        trainer = CDDRecTrainer(model, train_dataloader, eval_dataloader, test_dataloader, args)

    if args.do_eval:
        trainer.args.train_matrix = test_rating_matrix
        scores, result_info, _ = trainer.test('test', full_sort=True)

    else:

        early_stopping = EarlyStopping(args.checkpoint_path, patience=50, verbose=True)
        if args.load_model:
            trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        for epoch in range(args.epochs):
            start = time.time()
            trainer.train(epoch)
            # evaluate on MRR
            scores, _, _ = trainer.valid(epoch, full_sort=True)
            early_stopping(np.array([scores[4], scores[5]]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            end = time.time()
            print(end-start)
        print('---------------Change to test_rating_matrix!-------------------')
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        valid_scores, _, _ = trainer.valid('best', full_sort=True)
        trainer.args.train_matrix = test_rating_matrix
        scores, result_info, _ = trainer.test('best', full_sort=True)

    print(args_str)
    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')
main()

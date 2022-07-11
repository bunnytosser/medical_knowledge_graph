# -*- coding:utf-8 -*-
"""
@Author: Zhujia
@Date: 2021-07-21 10:00:03

主要功能：
        对单个语料（a:全量数据， m:操作手册 c：临床指南 kd：知识库 d：药品说明书）语料进行训练，生成模型并存储到相应的路径。
        可以从0开始训练，也可以对基础模型进行继续训练。
        最重要的参数包括：cp 语料类型，a:全量数据， m:操作手册 c：临床指南 kd：知识库 d：药品说明书
                        ag 是否使用数据增强，默认为0
                        l 是否使用继续学习，默认为0不适用，如果使用那么会从指定目录读取相应语料训练出的已有的模型继续训练。
                        其他参数查看代码详情。

使用场景：多语料模型分别训练。训练多个模型后可供后续模型评估，知识生产等工作。

"""
import pickle
import math
import sys
import os
import time
import yaml
import jieba
import json
import logging
import logging.config
from shutil import copyfile
import copy
import torch
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
import torch.nn.utils as utils
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from ner.cond_data_manager import DataManager
from ner.cond_model import BiLSTMCRF
from ner.utils import f1_score, f1_score_winning, get_tags, format_result, \
    check_n_update_tag_n_output_entity, check_tag_n_output_train_plus, muc_evaluation
from ner.predict import get_pred_tags
from kmeans_pytorch import kmeans,pairwise_cosine,pairwise_distance
torch.cuda.manual_seed(42)
from loguru import logger

import argparse
parser = argparse.ArgumentParser(description='Enigma Version PyTorch')

parser.add_argument('-cp', '--corpus', default = "m", type=str, help = 'which corpus to use for training')
#parser.add_argument('-tanh', '--tanh', default = 1, type=int, help = 'use tanh function to transform logits matrix')
parser.add_argument('-ag', '--augmentation', default = 0, type=str, help = 'whether to use augmented data or not')
parser.add_argument('-t', '--toy', default=0, type = int, help = 'whether or not to use a very small fraction of training samples to test the model')
parser.add_argument('-c','--conditional_type', default=0,type=int, help='使用何种LSTM条件，0为不使用，1为仅使用段落信息，2为使用段落信息和实体1信息,-1为隐藏状态使用段落信息')
parser.add_argument('-a','--active', default="None",type=str, help='是否启用主动学习模式继续训练: None 不采用 ctn 继续学习 ft fine-tuning')
parser.add_argument('-l','--continue_learning', default=0, type=int, help='是否要在上一次训练的模型基础上继续训练')
parser.add_argument('-b','--balanced', default=0, type=int, help='是否采用平衡的采样，以平衡个数据来源的样本数量')
parser.add_argument('-s','--seg_emb', default=2, type=int, help='字符嵌入方式 0：使用字嵌入，1：额外使用seq结巴分词信息 2：嵌入后还进行CNN的嵌入')
parser.add_argument('-w','--cws', default=0, type=int, help='whether to have a chinese word segmentation model with shared cnn-lstm layer co-trained with NER')

parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--decay', default=1e-8, type=float, help='weight decay')
parser.add_argument('--clip_max_norm', default=None, type=float, help='clip_max_norm for grad')
parser.add_argument('--eval_freq', default=50, type=int, help='多少个batch迭代执行一次dev')

parser.add_argument('--pred_candidates', default=True, type=bool, help='使用待标注数据做预测从而从中采样')
parser.add_argument('--test_freq', default=1500, type=int, help='多少次迭代执行一次test')

parser.add_argument('--seg_emb_dim', default=20, type=int, help='seq_emb_dim的维度')
parser.add_argument('--char_emb_dim', default=100, type=int, help='char_emb_dim的维度')

parser.add_argument('--entity_recognization', default=False, type=bool, help='是否要进行实体识别')
parser.add_argument('--er_amount', default=500, type=int, help='实体识别使用的数据量')
parser.add_argument('--er_freq', default=500, type=int, help='实体识别频率')

args = parser.parse_args()

logger.info("character embedding style:",args.seg_emb)


# tag_dict = {'O': 0,
#             'START': 1,
#             'STOP': 2,
#             'B-REG': 3, 'I-REG': 4, 'B-ORG': 5, 'I-ORG': 6, 'B-SGN': 7, 'I-SGN': 8, 'B-NBP': 9, 'I-NBP': 10,
#             'B-SYM': 11, 'I-SYM': 12, 'B-AT': 13, 'B-FW': 14, 'I-FW': 15, 'B-PT': 16, 'I-PT': 17, 'B-DIS': 18,
#             'I-DIS': 19, 'B-TES': 20, 'I-TES': 21, 'B-DRU': 22, 'I-DRU': 23, 'B-DEG': 24, 'I-DEG': 25, 'I-AT': 26,
#             'B-PRE': 27, 'I-PRE': 28, 'B-DUR': 29, 'I-DUR': 30, 'B-PRP': 31, 'I-PRP': 32, 'B-BFL': 33, 'B-SUR': 34,
#             'I-SUR': 35, 'B-PSB': 36, 'I-PSB': 37, 'I-BFL': 38}

# tag_dict_2 = {'O': 0, 'START': 1, 'STOP': 2,
#                'B-REG': 3, 'I-REG': 4, 'B-ORG': 5, 'I-ORG': 6, 'B-SGN': 7, 'I-SGN': 8,
#                'B-SYM': 9, 'I-SYM': 10, 'B-AT': 11, 'B-FW': 12, 'I-FW': 13, 'B-DIS': 14, 'I-DIS': 15,
#                'B-PT': 16, 'I-PT': 17, 'B-TES': 18, 'I-TES': 19, 'B-DRU': 20, 'I-DRU': 21,
#                'B-DEG': 22, 'I-DEG': 23, 'B-DUR': 24, 'I-DUR': 25, 'I-AT': 26, 'B-PRE': 27, 'I-PRE': 28,
#                'B-PRP': 29, 'I-PRP': 30, 'B-BFL': 31, 'I-BFL': 32, 'B-SUR': 33, 'I-SUR': 34,
#                'B-PSB': 35, 'I-PSB': 36}

# tag_dict_bac_mat = {'O': 0, 'START': 1, 'STOP': 2, 'B-PRP': 3, 'I-PRP': 4, 'B-DRU': 5, 'I-DRU': 6,
#                     'B-SGN': 7, 'I-SGN': 8, 'B-DIS': 9, 'I-DIS': 10, 'B-SYM': 11, 'I-SYM': 12, 'B-BAC': 13, 'I-BAC': 14,
#                     'B-AT': 15, 'B-SUR': 16, 'I-SUR': 17, 'B-ORG': 18, 'I-ORG': 19, 'B-DEG': 20, 'I-DEG': 21,
#                     'B-PRE': 22, 'I-PRE': 23, 'B-TES': 24, 'I-TES': 25, 'I-AT': 26, 'B-PSB': 27, 'I-PSB': 28,
#                     'B-BFL': 29, 'I-BFL': 30, 'B-DUR': 31, 'I-DUR': 32, 'B-REG': 33, 'I-REG': 34,
#                     'B-PT': 35, 'I-PT': 36, 'B-MAT': 37, 'I-MAT': 38, 'B-FW': 39, 'I-FW': 40}




AUG_CUMULATE_CNT = 0

def sigmoid_customized(x):
    return 1 / (1 + (10/(x-2)**2 ))

def setup_logging(default_path="ner/logs/config/logging.yaml", default_level=logging.INFO):
    path = default_path

    if os.path.exists(path):
        with open(path, "r") as f:
            config = yaml.load(f)
            timestamps_str = str(int(time.time()))
            config['handlers']['info_file_handler']['filename'] = 'ner/logs/info_'+timestamps_str + '.log'
            config['handlers']['error_file_handler']['filename'] = 'ner/logs/errors_' + timestamps_str + '.log'
            config['handlers']['result_file_handler']['filename'] = 'ner/logs/results_' + timestamps_str + '.log'
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

class DAL(nn.Module):
    def __init__(self):
        super(DAL,self).__init__()
        self.fc1 = nn.Linear(156,20)
        self.fc2 = nn.Linear(20,2)
        self.sm = nn.Softmax()
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
#        x=selfg.sm(x,dim=-1)
        return x


class ChineseNER(object):
    def __init__(self, entry="train", model_path_name = "ner/models/model_2022/params_o_cnn_a.pkl"):
        rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        logfile = 'logs/ner_model_{}_{}_{}.log'.format(entry,args.corpus,rq)
        logger.add(logfile, backtrace=True, diagnose=True, rotation='3 days', retention='2 months')
        logger.info("STAGE 0: Initializing ")
        with open('config.yaml', 'rb') as fp:
            dic_file = yaml.safe_load(fp)
            essential_dic = dic_file['ontologies']['RE']
            nonessential_dic = dic_file['ontologies']['RE_nonessential']
            self.tag_mapping = {**essential_dic, **nonessential_dic}
            ner_training_final = dic_file['training_data']['ner_training_final']
            self.model_folder = dic_file['model_path']

#         print('tag_mapping!!!!',self.tag_mapping)
        self.parampath = "{}/params_o_cnn_{}_params.pkl".format(self.model_folder, args.corpus)
        if args.augmentation == 1:
            self.TRAIN_PATH = "{}/augmented_{}_training.txt".format(ner_training_final, args.corpus)  
        else:
            self.TRAIN_PATH = "{}/{}_training.txt".format(ner_training_final, args.corpus)
        self.DEV_PATH = "{}/{}_validation.txt".format(ner_training_final, args.corpus)
        self.TEST_PATH = "{}/test_seed.txt".format(ner_training_final)
        self.mode = entry
        self.load_config()
        self.labeloutput = "ner/data/labeling_output.pickle"
        model_path = model_path_name
        # model_path = '{}/params_o_cnn_a.pkl'.format(self.model_folder)
        self.__init_model(entry, model_path)
        self.new_entity_tag_dict_path = '{}/dict/new_dics.json'.format(self.model_folder)
        #self.conditional_type=args.conditional_type
        with open(self.new_entity_tag_dict_path, 'r', encoding='utf8') as fp:
            self.new_entity_tag_dict = json.load(fp)
        setup_logging()

    def __init_model(self, entry, model_path):
        self.cuda_available = torch.cuda.is_available()
        self.conditional_type = args.conditional_type
        self.cws = args.cws
        self.modelpath = model_path
        if entry == 'predictonly':
            print("=====predict only====") 
            self.modelpath = model_path
        else:
#            if tanh == 1:
#                args.corpus = args.corpus + "_v2"
            logger.info("corpus name", args.corpus)
            if self.cws == 1:
                self.modelpath = "{}/params_cws_{}.pkl".format(self.model_folder, args.corpus)
            elif args.active == "None" and args.seg_emb == 2:
                logger.info("{}: {}".format(type(args.augmentation),args.augmentation))

                if args.augmentation == 0:
                    self.modelpath = "{}/params_o_cnn_{}.pkl".format(self.model_folder, args.corpus)
                elif args.augmentation == 1:
                    logger.info("generating model name")
                    self.modelpath = "{}/params_o_cnn_{}_aug.pkl".format(self.model_folder, args.corpus)
            elif args.active == "None" and args.seg_emb != 2:
                if args.augmentation == 0:
                    self.modelpath = "{}/params_o_{}.pkl".format(self.model_folder, args.corpus)
                elif args.augmentation == 1:
                    self.modelpath = "{}/params_o_{}_aug.pkl".format(self.model_folder, args.corpus)
            elif args.active != "None" and args.seg_emb != 2:
                self.modelpath = "{}/active_params_{}.pkl".format(self.model_folder, args.corpus)
            elif args.active != "None" and args.seg_emb == 2:
                if args.augmentation == 0:
                    self.modelpath = "{}/active_params_cnn_v2_{}.pkl".format(self.model_folder, args.corpus)
                else:
                    self.modelpath = "{}/active_params_cnn_v2_{}_aug.pkl".format(self.model_folder, args.corpus)
        ## 保存该其他参数的pickle
        self.parampath = self.modelpath.split(".")[0] + "_params.pkl"  
        print("=====path====", self.parampath)     
        # 加载jieba字典
#        self.jieba_entity_dict_txt_path = 'models/dict/jieba_entity_dict.txt'
#        jieba.load_userdict(self.jieba_entity_dict_txt_path)
        # print('jieba userdict loaded!')

        if entry == "train":
            """ ==== 读取训练数据 ====  """
            # print("=====current tags",self.tags)
            logger.info('读取训练数据')
            # print("=====toy ",args.toy)
            self.train_manager = DataManager(batch_size=self.batch_size, data_path=self.TRAIN_PATH, tags=self.tags, toy=args.toy)

            with open('ner/data/tag_map.json', 'w') as fp:
                json.dump(self.train_manager.tag_map,fp)

            self.total_size = len(self.train_manager.batch_data)
            data = {
                "batch_size": self.train_manager.batch_size,
                "input_size": self.train_manager.input_size,
                "vocab": self.train_manager.vocab,
                "tag_map": self.train_manager.tag_map,
            }
            self.save_params(data)

            """ ==== 读取验证数据 ====  """
            logger.info('读取验证数据')
            dev_manager = DataManager(batch_size=self.batch_size, data_path=self.DEV_PATH)
#            print("-------tag map:",self.train_manager.tag_map)
            self.dev_batch = dev_manager.iteration()


            """ ==== 读取测试数据 ====  """
            if args.pred_candidates:
                logger.info('读取待标注数据')
                test_manager = DataManager(batch_size=self.batch_size, data_path=self.TEST_PATH)
                self.test_batch = test_manager.iteration()
                self.test_data = test_manager.data
                

            """ ==== 加载模型结构 ====  """
            self.model = BiLSTMCRF(
                tag_map = self.train_manager.tag_map,
  #              ent_map = self.train_manager.entities,
                batch_size = self.batch_size,
                vocab_size = len(self.train_manager.vocab),
                dropout = self.dropout,
                char_emb_size = self.char_emb_size,
                seg_emb_size = self.seg_emb_size,
                hidden_dim = self.hidden_size,
                seg_emb = args.seg_emb,
                cuda_available = self.cuda_available,
                conditional_type = self.conditional_type,
                cws = self.cws

            )

            """ ==== 加载模型参数 ====  """
            if args.continue_learning == 1:
                self.restore_model(self.modelpath)

        elif entry == "predict":
            data_map = self.load_params()
            input_size = data_map.get("input_size")
            self.tag_map = data_map.get("tag_map")
            self.vocab = data_map.get("vocab")

  #          self.ent_map = data_map.get("entities")

            test_manager = DataManager(batch_size=self.batch_size, data_path=self.TEST_PATH)
            self.test_batch = test_manager.iteration()
            self.test_data = test_manager.data
            dev_manager = DataManager(batch_size=self.batch_size, data_path=self.DEV_PATH)
            self.dev_batch = dev_manager.iteration()
            self.dev_data = dev_manager.data

            self.model = BiLSTMCRF(
                tag_map = test_manager.tag_map,
#                ent_map = test_manager.entities,
                batch_size = self.batch_size,
                vocab_size = input_size,
                char_emb_size = self.char_emb_size,
                seg_emb_size = self.seg_emb_size,
                hidden_dim = self.hidden_size,
                seg_emb = args.seg_emb,
                cuda_available = self.cuda_available,
                conditional_type = self.conditional_type,
                cws = self.cws
            )
            #self.restore_model("params.pkl")

        elif entry == "predictonly":
            data_map = self.load_params()
            input_size = data_map.get("input_size")
            self.tag_map = data_map.get("tag_map")
            self.vocab = data_map.get("vocab")
 
            self.model = BiLSTMCRF(
                tag_map = self.tag_map,
                batch_size = self.batch_size,
                vocab_size = input_size,
                char_emb_size = self.char_emb_size,
                seg_emb_size = self.seg_emb_size,
                hidden_dim = self.hidden_size,
                seg_emb = args.seg_emb,
                cuda_available = self.cuda_available,
                conditional_type = self.conditional_type,
                cws = self.cws
            )
            self.restore_model(self.modelpath)


    def load_config(self):
        try:
            fopen = open("{}/config-winning.yml".format(self.model_folder))
            config = yaml.load(fopen)
            fopen.close()
        except Exception as error:
            logger.info("Load config failed, using default config {}".format(error))
            fopen = open("{}/config-winning.yml".format(self.model_folder), "w")

            config = {
                "embedding_size": 100,
                "char_emb_size": 100,
                "seg_emb_size": 20,
                "hidden_size": 128,
                "batch_size": 20,
                "dropout": 0.5,
                "model_path": "models/",
                "tags": ["ORG", "PER"]
            }
            yaml.dump(config, fopen)
            fopen.close()
        self.char_emb_size = 50
        self.seg_emb_size = 0
        if args.seg_emb:
            self.char_emb_size = config.get("char_emb_size")
            self.seg_emb_size = config.get("seg_emb_size")
        else:
            self.char_emb_size = config.get("char_emb_size")
        self.hidden_size = config.get("hidden_size")
        self.batch_size = config.get("batch_size")
        self.model_path = config.get("model_path")
        self.tags = config.get("tags")
        self.dropout = config.get("dropout")

    def restore_model(self,modelname):
        try:
            self.model.load_state_dict(torch.load(modelname), strict=False)
           # self.model.load_state_dict(torch.load(self.model_path + modelname), strict=False)
            logger.info("model [%s] restore success!"%modelname)
        except Exception as error:
            logger.info("model {} restore faild! {}".format(modelname,error))

    def save_params(self, data):

        with open(self.parampath, "wb") as fopen:
            pickle.dump(data, fopen)

    def load_params(self):
        with open(self.parampath, "rb") as fopen:
            data_map = pickle.load(fopen)
        return data_map

    def pad_data(self, data,padcontent=0):
        max_length = max([len(i) for i in data])
        i_idx = 0
        error_idx = []
        c_data = []
        for i in data:
            # i是list，包含一个sentence中的信息，包括char_id_list，tag_id_list, seg_info_list
            # 现在加上sentence长度信息
            if len(i) == 0:
                error_idx.append(i_idx)
                logger.info('remove one')
                continue

            # sentence补齐。char_id用0补齐，tag_id用0补齐，seg_id用4补齐（0-3已有含义）
            i = i+ (max_length - len(i)) * [padcontent]
            c_data.append(i)
            i_idx += 1
        # a, b = zip(*sorted(zip(a, b)))
        rm_cnt = 0
        if len(error_idx) > 0:
            for to_rm_idx in error_idx:
                c_data.pop(to_rm_idx - rm_cnt)
                rm_cnt += 1
        return c_data

    def train(self,active=args.active):
        # 梯度裁剪
        if args.clip_max_norm is not None:
            utils.clip_grad_norm(self.model.parameters(), max_norm=args.clip_max_norm)
        ttl_index = 0
        epoch_index=0

        if active == "ctn" or active == "ft":

            self.restore_model(self.modelpath)
            self.model.train()
            device_gpu = torch.device('cuda:0')
            self.model = self.model.cuda(device_gpu)

            if mode == "ctn":
                for param in self.model.parameters():
                    param.requires_grad=True
            elif mode == "ft":
                for param in self.model.parameters():
                    param.requires_grad = False
                if self.cuda_available: 
                    self.model.transitions = nn.Parameter(torch.randn(self.model.tag_size,self.model.tag_size)).cuda() 
                else: 
                    self.model.transitions = nn.Parameter(torch.randn(self.model.tag_size,self.model.tag_size)) 

            logger.info("-"*20,"%s total numer of params under mode %s:"%(sum(p.numel() for p in self.model.parameters()),mode))
            logger.info("-"*20,"%s total numer of trainables under mode %s:"%(sum(p.numel() for p in self.model.parameters() if p.requires_grad),mode))

            optimizer = optim.Adam(self.model.parameters(), lr=(args.lr)/5, weight_decay=args.decay)
            for epoch in np.arange(1):
                epoch_index += 1
                logger.info('STEP%s: '%epoch, '=' * 120)
                ###training under active learning mode

                with open(self.labeloutput, 'rb') as inp:
                    sentences = pickle.load(inp)
                    tags = pickle.load(inp)
                    seg_infos = pickle.load(inp)
                    char_sentence = pickle.load(inp)
      #              source=pickle.load(inp)
      #              entity=pickle.load(inp)
                    length = pickle.load(inp)
                
                if len(sentences) <= 50:
                    sentences = sentences*30
                    tags = tags*30
                    seg_infos = seg_infos*30
                    char_sentence = char_sentence*30
     #               source=source*30
     #               entity=entity*30
                    length = length*30
                sentences = self.pad_data(sentences,0)
                tags = self.pad_data(tags,0)
                seg_infos = self.pad_data(seg_infos,4)
                for pseudo_batch in [0]:
                #   self.model.zero_grad()
                    optimizer.zero_grad()
                    current_batch_size = len(sentences)
                    if self.cuda_available:
                        sentences_tensor = torch.tensor(sentences, dtype=torch.long).cuda()
                        tags_tensor = torch.tensor(tags, dtype=torch.long).cuda()
                        seg_infos_tensor = torch.tensor(seg_infos, dtype=torch.long).cuda()
                        length_tensor = torch.tensor(length, dtype=torch.long).cuda()

                    else:
                        sentences_tensor = torch.tensor(sentences, dtype=torch.long)
                        tags_tensor = torch.tensor(tags, dtype=torch.long)
                        seg_infos_tensor = torch.tensor(seg_infos, dtype=torch.long)
                        length_tensor = torch.tensor(length, dtype=torch.long)


                    loss = self.model.neg_log_likelihood(sentences_tensor, tags_tensor, seg_infos_tensor, length_tensor, current_batch_size)
#                    if ttl_index % 50 == 0:
#                        print('ttl_index: ', ttl_index)
#                        progress = ("█" * int(epoch_index * 25 / self.total_size)).ljust(25)
#                        print("""epoch [{}] |{}| {}/{}\n\tloss {:.2f}""".format(
#                        epoch, progress, epoch_index, self.total_size, loss.cpu().tolist()[0]
#                    )
#                    )
#
                    loss.backward()
                    optimizer.step()

                # 进行模型dev评估
                #print("-"*20+"start evaluation:"+"-"*20)
                #self.evaluate(mode='dev')
                logger.info("-" * 50)
                torch.save(self.model.state_dict(), self.modelpath)
                logger.info("model saved successfully")
           
            # train the initial NER model
        else:
            self.model.train()
            device_gpu = torch.device('cuda:0')
            self.model = self.model.cuda(device_gpu)


            optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.decay)
            for epoch in np.arange(5):
                epoch_index += 1
                logger.info("STEP %s"%epoch,"="*200)
                for batch in self.train_manager.get_batch():
                    ttl_index += 1
                #    if ttl_index>=40:
                #        break
                # self.model.zero_grad()
                    optimizer.zero_grad()
                    logger.info(zip(*batch))
                    sentences, tags, seg_infos, char_sentence, length = zip(*batch)
                    current_batch_size = len(sentences)
                    if self.cuda_available:
                        sentences_tensor = torch.tensor(sentences, dtype=torch.long).cuda()
                        tags_tensor = torch.tensor(tags, dtype=torch.long).cuda()
                        seg_infos_tensor = torch.tensor(seg_infos, dtype=torch.long).cuda()

                        length_tensor = torch.tensor(length, dtype=torch.long).cuda()
                 #       entity_tensor = torch.tensor(entity, dtype=torch.long).cuda()
                 #       source_tensor = torch.tensor(source, dtype=torch.long).cuda()
                    else:
                        sentences_tensor = torch.tensor(sentences, dtype=torch.long)
                        tags_tensor = torch.tensor(tags, dtype=torch.long)
                        seg_infos_tensor = torch.tensor(seg_infos, dtype=torch.long)
                        length_tensor = torch.tensor(length, dtype=torch.long)
                 #       entity_tensor = torch.tensor(entity, dtype=torch.long)
                 #       source_tensor = torch.tensor(source, dtype=torch.long)

                    #print("shapes:",entity_tensor.shape,source_tensor.shape)

                    loss = self.model.neg_log_likelihood(sentences_tensor, tags_tensor, seg_infos_tensor, length_tensor,
                                                     current_batch_size)

                    if ttl_index % 50 == 0:
                        logger.info('ttl_index: ', ttl_index)
                        progress = ("█" * int(epoch_index * 25 / self.total_size)).ljust(25)
                        logger.info("""epoch [{}] |{}| {}/{}\n\tloss {:.2f}""".format(
                        epoch, progress, epoch_index, self.total_size, loss.cpu().tolist()[0]
                    )
                    )

                    loss.backward()
                    optimizer.step()

                # 进行模型dev评估
                    if ttl_index % args.eval_freq == 0 and ttl_index != 0 :
                        if args.pred_candidates and ttl_index % args.test_freq == 0:
                            logger.info("-"*20+"start testing evaluation:"+"-"*20)
                            self.evaluate(mode='test')
                #            except:
                #                print(" evaluation failed")
                        else:
                            self.evaluate(mode='dev')
                            print("-"*20 + "start validation evaluation:"+"-"*20)
                #            except:
                #                print(" evaluation failed")
                        logger.info("-" * 50)
                        torch.save(self.model.state_dict(), self.modelpath)

                        #torch.save(self.model, self.model_path + 'model.pkl')

                    # 进行实体数据识别
                    if args.entity_recognization and ttl_index % args.er_freq == 0:
                        self.entity_recognization()
                        logger.info("-" * 50)


    def pred(self):
        logger.info("**"*20,"predicting")
        paths_dev = []
        labels_dev = []
        length_dev = []
        dev_batch_cnt = 0
        dev_precision = 0
        dev_recall = 0
        dev_f1 = 0
        dev_cnt = 0
        pred_batch_counter = 0
        pred_scores = []
        vit_scores = []
        lstm_outputs = []
        all_lengths = []
        ltps = []
        while True:
            self.model.eval()
            sentences, labels, seg_infos, char_sentences, lengths = zip(*self.test_batch.__next__())
            dev_batch_cnt += 1
            # print(dev_batch_cnt)
            #print(sentences)
             ## the last batch does not count if its incomplete
            if dev_batch_cnt > math.floor(len(self.test_data)/self.batch_size):
                break
            cut_short_labels = []
            for label, length in zip(labels, lengths):
                cut_short_labels.append(label[:length])
                all_lengths.append(length)

            if self.cuda_available:
                sentences_tensor = torch.tensor(sentences, dtype=torch.long).cuda()
                tags_tensor = torch.tensor(labels, dtype=torch.long).cuda()
                seg_infos_tensor = torch.tensor(seg_infos, dtype=torch.long).cuda()
                length_tensor = torch.tensor(lengths, dtype=torch.long).cuda()
            #    entity_tensor = torch.tensor(entity, dtype=torch.long).cuda()
            #    source_tensor = torch.tensor(source, dtype=torch.long).cuda()
            else:
                sentences_tensor = torch.tensor(sentences, dtype=torch.long)
                tags_tensor = torch.tensor(labels, dtype=torch.long)
                seg_infos_tensor = torch.tensor(seg_infos, dtype=torch.long)
                length_tensor = torch.tensor(lengths, dtype=torch.long)

            current_batch_size = len(sentences)
            #ll_score=self.model.log_likelihood(sentences_tensor, tags_tensor, seg_infos_tensor, length_tensor,current_batch_size)
            scores, paths, ltp, p_probs = self.model(sentences_tensor, seg_infos_tensor, length_tensor)
            #print("="*20,"scores",scores)
               # sentences, tags, seg_infos, char_sentence, length = zip(*batch)
               # current_batch_size = len(sentences)
            lstm_outputs.append(self.model.lstm_out)
            ltps.append(ltp)

            vit_scores.append(scores)
            labels_dev.extend(list(cut_short_labels))
            paths_dev.extend(list(paths))

            length_dev.extend(list(lengths))


            pred_batch_counter += 1
        return(vit_scores,all_lengths,ltps)


    def evaluate(self, mode='dev'):
        paths_dev = []
        labels_dev = []
        length_dev = []
        dev_batch_cnt = 0
        # dev_precision = 0
        # dev_recall = 0
        # dev_f1 = 0
        # dev_cnt = 0
        while True:
            self.model.eval()
            logger.info(zip(*self.dev_batch.__next__()))

            if mode == 'dev':
                sentences, labels, seg_infos, char_sentences, lengths = zip(*self.dev_batch.__next__())
            elif mode == 'test':
                sentences, labels, seg_infos, char_sentences, lengths = zip(*self.test_batch.__next__())
            else:
                raise Exception('mode error:', mode)
            dev_batch_cnt += 1
            # print(dev_batch_cnt)
            # print(sentences)
            if dev_batch_cnt > 10:
                break
            cut_short_labels = []
            for label, length in zip(labels, lengths):
                cut_short_labels.append(label[:length])

        #    try:
            if self.cuda_available:
                sentences_tensor = torch.tensor(sentences, dtype=torch.long).cuda()
                tags_tensor = torch.tensor(labels, dtype=torch.long).cuda()
                seg_infos_tensor = torch.tensor(seg_infos, dtype=torch.long).cuda()
                length_tensor = torch.tensor(lengths, dtype=torch.long).cuda()
            #    entity_tensor = torch.tensor(entity, dtype=torch.long).cuda()
            #    source_tensor = torch.tensor(source, dtype=torch.long).cuda()
            else:
                sentences_tensor = torch.tensor(sentences, dtype=torch.long)
                tags_tensor = torch.tensor(labels, dtype=torch.long)
                seg_infos_tensor = torch.tensor(seg_infos, dtype=torch.long)
                length_tensor = torch.tensor(lengths, dtype=torch.long)

            current_batch_size = len(sentences)
            #ll_score=self.model.log_likelihood(sentences_tensor, tags_tensor, seg_infos_tensor, length_tensor,current_batch_size)
            scores, paths, ltp, p_probs = self.model(sentences_tensor, seg_infos_tensor, length_tensor)

            # scores, paths, ltps, p_probs = self.model(sentences, seg_infos, lengths)

            labels_dev.extend(list(cut_short_labels))
            paths_dev.extend(list(paths))
            length_dev.extend(list(lengths))

        results, results_agg = muc_evaluation(labels_dev, paths_dev, length_dev, self.model.tag_map)

        self.model.train()

    def entity_recognization(self):
        paths_dev = []
        labels_dev = []
        length_dev = []
        char_sentence_dev = []
        aug_batch_cnt = 0
        dev_precision = 0
        dev_recall = 0
        dev_f1 = 0
        dev_cnt = 0
        logger.info("\tentity_recognization start")
        while True:
            self.model.eval()
            # 这里暂时有labels，但也要准备没有label的情况
            sentences, labels, seg_infos, char_sentences, lengths = zip(*self.aug_batch.__next__())
            aug_batch_cnt += 1
            if aug_batch_cnt > 50:
                break
            cut_short_labels = []
            for label, length in zip(labels, lengths):
                cut_short_labels.append(label[:length])
            _, paths,ltps, p_paths = self.model(sentences, seg_infos, lengths)
            char_sentence_dev.extend(list(char_sentences))
            labels_dev.extend(list(cut_short_labels))
            paths_dev.extend(list(paths))
            length_dev.extend(list(lengths))
        logger.info("\tentity_recognization step1 finished")
        # print(self.model.tag_map)

        distinct_tags = set(map(lambda tag: tag.replace('B-', '').replace('I-', '').replace('E-', ''), self.tags))
        new_entity_dict = check_n_update_tag_n_output_entity(char_sentence_dev, paths_dev, length_dev,
                                                                 distinct_tags, self.model.tag_map,
                                                                 self.new_entity_tag_dict)

        logger.info('result_dict:\n', new_entity_dict)
        with open('{}/new_find_dict.json'.format(self.model_folder), 'w', encoding='utf8') as fp:
            json.dump(new_entity_dict, fp, ensure_ascii=False)

        logger.info("\tentity_recognization step2 finished")
        self.model.train()


    def predict(self, input_str=""):
        if not input_str:
            input_str = input("请输入文本: ")
        input_vec = [self.vocab.get(i, 0) for i in input_str]
        # convert to tensor
        sentences = torch.tensor(input_vec).view(1, -1)
        _, paths,ltps, p_paths = self.model(sentences)

        entities = []
        for tag in self.tags:
            tags = get_tags(paths[0], tag, self.tag_map)
            entities += format_result(tags, input_str, tag)
        return entities

    def get_seg_features(self, string):
        """
        Segment text with jieba
        features are represented in bies format
        s donates single word
        """
        seg_feature = []

        for word in jieba.cut(string, cut_all=False):
            if len(word) == 1:
                seg_feature.append(0)
            else:
                tmp = [2] * len(word)
                tmp[0] = 1
                # tmp[-1] = 3
                seg_feature.extend(tmp)
        return seg_feature
    
    def predict_multiline(self, input_str,source):
        inverse_tag_map = dict([val, key] for key, val in self.tag_map.items())
        #logging.info('inverse_tag_map:{}'.format(str(inverse_tag_map)))
        #logging.info('get input:{}'.format(input_str))
        lengths = []
        seg_infos = []
        input_vecs = []
        char_lists = []

        for s in input_str:
            char_list = list(s)
            length = len(s)
        #    self.model.eval()
            seg_info = self.get_seg_features(s)
            input_vec = [self.vocab.get(i, 0) for i in s]
            lengths.append(length)
            input_vecs.append(input_vec)
            seg_infos.append(seg_info)
            char_lists.append(list(s))
        # convert to tensor
        input_vecs = self.pad_data(input_vecs,0)
        seg_infos = self.pad_data(seg_infos,4)

        length = torch.tensor(lengths).view(len(input_str), -1)
        sentences = torch.tensor(input_vecs).view(len(input_str), -1)
        seg_infos = torch.tensor(seg_infos).view(len(input_str), -1)

        if self.cuda_available:
            # sentences = torch.tensor(sentences, dtype=torch.long).cuda()
            sentences = sentences.cuda()
            seg_infos = torch.tensor(seg_infos, dtype=torch.long).cuda()
            length = torch.tensor(length, dtype=torch.long).cuda()
            device_gpu = torch.device('cuda:0')
            self.model = self.model.cuda(device_gpu)

        scores, paths, ltp, p_paths = self.model(sentences, seg_infos, length)
        result_dicts = []
        for char_list, path, length in zip(char_lists,paths,lengths):
            result_dict, result_cnt = get_pred_tags(char_list, path, length, inverse_tag_map)
            for key, value in result_dict.items():
                result_dict[key] = [result_dict[key][0], result_dict[key][1], self.tag_mapping[result_dict[key][1]], result_dict[key][2]]
            result_dicts.append(result_dict)
        #logging.info('result_dict:{}'.format(str(result_dict)))
        #logging.info('=' * 300)
        return result_dicts

    def predict_oneline(self, input_str='没有输入文本',  explanation=False):

        inverse_tag_map = dict([val, key] for key, val in self.tag_map.items())
        #logging.info('inverse_tag_map:{}'.format(str(inverse_tag_map)))
        #logging.info('get input:{}'.format(input_str))
        char_list = list(input_str)
        length = [len(input_str)]
        self.model.eval()
        seg_infos = self.get_seg_features(input_str)

        input_vec = [self.vocab.get(i, 0) for i in input_str]
    #    print('input_vec: ', input_vec)

        # convert to tensor
        sentences = torch.tensor(input_vec).view(1, -1)

        if self.cuda_available:
            # sentences = torch.tensor(sentences, dtype=torch.long).cuda()
            sentences = sentences.cuda()
            seg_infos = torch.tensor(seg_infos, dtype=torch.long).cuda()
            length = torch.tensor(length, dtype=torch.long).cuda()
            device_gpu = torch.device('cuda:0')
            self.model = self.model.cuda(device_gpu)

        scores, paths, ltp, path_probs = self.model(sentences, seg_infos, length)
        path_probs_2 = []
        for pb in path_probs[0]:
            pb = sigmoid_customized(pb)
            path_probs_2.append(pb)
     #   print(scores, paths, ltp, path_probs)
        result_dict, result_cnt = get_pred_tags(char_list, paths[0], length, inverse_tag_map)

        for key, value in result_dict.items():
            ## 对输出的数值进行一定 自定义的sigmoid函数映射，成概率形式。取预测词边界内最大概率者 作为预测的‘模糊边界’准确率，最低概率者作为 严格准确率
            word_probs = path_probs_2[result_dict[key][2][0] : result_dict[key][2][1] + 1]
          #  print(word_probs, result_dict[key][0])
            avg_prob_max = np.max(word_probs)
            avg_prob_min = np.min(word_probs)
#             print("！！！！！！！！！！！！tag mapping", self.tag_mapping)
#             print("！！！！！！！！！！！！result_dict", key, result_dict[key])
            result_dict[key] = [result_dict[key][0], result_dict[key][1], self.tag_mapping[result_dict[key][1]], result_dict[key][2], [avg_prob_min, avg_prob_max]]
        return result_dict, path_probs_2

 

if __name__ == "__main__":

    if args.active == "ctn" or args.active == "ft":
        mode = "predict"
    else:
        mode = 'train'
    if os.path.exists("ner/data/labeling_output.pickle") is False:
        test_manager = DataManager(batch_size=64, data_path=self.TEST_PATH)
        test_data = test_manager.data
        selected_indices = np.random.choice(len(test_data), 100)
        LC_selected_data = [test_data[i] for i in selected_indices]                    

    else:
        cn = ChineseNER(mode)
        train_result = cn.train(args.active)

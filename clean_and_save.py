#!/usr/bin/env python
# -*- coding: utf-8 -*-



"""

__author__ = "朱佳"
__maintainer__ = "朱佳"
__create_date__ = "29/10/2021"

主要功能：标注后的后续处理工作。包括：ner结果与字典合并，依靠规则合并实体，清洗，各个结果表和中间表的存储，各个来源知识整合，最终存储到数据库。
            本代码包含以下内容
          * 字典匹配内容存储到结果表
          * 结果表字典和NER ensemble合并
          * 按照规则扩展实体边界
          * 存储和置信度
          * 临床指南结果整理，整合
          * 最后清理合并
          * 存储写入
          * NER RESULT规则清洗，存储，头实体存储
使用场景：知识生产。

本代码不包含：
1. NER模型训练，预测，标注（除临床指南的数据）
2. 临床指南的NER结果得到
"""

import pandas as pd
import os
import re
import requests
import numpy as np
import pandas as pd
import ast
import json
import jieba
import time
import jieba.posseg
from collections import defaultdict
import seaborn as sns
import copy
from clickhouse_driver import Client
from ner.ner_evaluation.ner_eval import Evaluator
import matplotlib.pyplot as plt
import yaml 
from loguru import logger
from datetime import datetime
import platform



                    
def in_between_inspector(rs, tag="v"):
    for dic in rs:
        if tag in dic["tags"]:
            print(dic)

def is_a_in_x(A, X):
    for i in range(len(X) - len(A) + 1):
        if A == X[i:i+len(A)]: return (i,True)
    return (0,False)

## 合并符合在 po_series 序列顺序的词为一个实体并存储，返回更新的全量数据
def words_combiner(merged_results,po_series):
# po_series=set(po_series)
    indications_copy = copy.deepcopy(merged_results)
    newly_merged = defaultdict(list)
    for j in indications_copy:
        j_dict = {}
        for z1,z2 in zip(list(range(0,len(j["seg"]))),j["seg"]):
            j_dict[z1] = z2
        try:
            segs = [s[1] for s in j["seg"]]
            words = [s[0] for s in j["seg"]]
        except:
            print("words combiner error", j["seg"])
        for seq in po_series:
            # iterating through all candidate sequences
            seqkey="|".join(seq)
    #         newly_merged[seqkey].append("h")
            (pos,contains)=is_a_in_x(seq,segs)
            if contains: 
                start = j["seg"][pos-1][2][1]
                combined = "".join(words[pos:pos+len(seq)])
                #限制长度
                if len(combined) >= 12:
                    continue
                if "、" in combined or combined not in j["sentence"] or "," in combined or ":" in combined:
                    continue
                print(combined, seq)
                newly_merged[seqkey].append(combined)
#                     j["seg"][pos:pos+len(seq)]=[[combined,seq[-1],[start,start+len(combined)]]]
                j_dict[pos] = [combined,seq[-1],[start,start+len(combined)]]
                try:
                    del j_dict[pos + 1]
                except:
                    continue
                if len(seq) == 3:
                    try:
                        del j_dict[pos+1]
                    except:
                        continue


        j["seg"] = list(j_dict.values())
    lennew = 0
    for j in newly_merged.values():
        lennew += len(j)
    print("newly discovered combinations:",lennew) 
    return(indications_copy)
#         newly_merged.append(new_words)
def merge_nerseg(indications):
    """
    先把NER和分词结果合并
    """
    global nonessential_dic, essential_dic, jieba_inverted
    notin_types = nonessential_dic
    desired_types = essential_dic
    jieba_dic = {v:k for (k,v) in jieba_inverted.items()}

    # getting the suffix for diseases and symptoms

    dissuf = []
    for k,v in suffix_dic.items():
        if v in desired_types:
            dissuf.append(k)     
    ## specify desired entity types and load suffix dictionaries
    ## merge the results for NER and dictionaries
    new_entities = []
    merged_results = []
    dicindex = 0
    for ix, dic in enumerate(indications):
        seg_info = dic["seg"]
        candidates_seg = [i for i in seg_info if i[1] in desired_types]
        candidates_ner = [i for i in dic["ensemble"] if i[1] in desired_types]
    #     print(candidates_seg,candidates_ner)
        for ner in candidates_ner:
            if "、" in ner[0] or "；" in ner[0] or "，" in ner[0] or ";" in ner[0]:
                continue
    #         if ner[0].startswith("、"):  # remove the "、" identified by NER
    #             ner[0]=ner[0][1:]
    #             ner[2]=[ner[2][0],ner[2][1]]
            ner_start = ner[2][0]
            ner_end = ner[2][1]
            ner_pos = [ner_start,ner_end]
            if ner_pos in [j[2] for j in candidates_seg]:
                continue
            elif len(ner[0]) > 2:
                seg0=[it[2][0] for it in seg_info]
                seg1=[it[2][1] for it in seg_info]
                if ner_start in seg0 and ner_end in seg1:
                    merge_0 = seg0.index(ner_start)
                    merge_1 = seg1.index(ner_end)
                    if merge_0 == merge_1: ##same entity boundary but entity type is different, since we trust dictionary more, 
                        # these results are discarded
                        continue
                    new_entities.append(ner)
                    for to_del in range(merge_1 + 1, merge_0, -1): ## change
    #                     print(ix,seg_info[to_del-1][:2])
                        seg_info.pop(to_del-1)
    #                 ner[2][1]=ner[2][1]+1
                    seg_info.insert(merge_0,ner)
                else:
                    pass
        dicindex += 1
        dic["ind"] = dic["ind"]
        
        for si,s in enumerate(seg_info):
            seg_info[si][1] = jieba_dic.get(s[1],s[1])

        dic["seg"] = seg_info
        merged_results.append(dic)
    return(merged_results)
    """分两轮，分别对尸体合并，后缀合并，再尸体合并"""
## 合并符合在指定位置前2个词内出现指定词性/实体的词，就进行合并并且返回更新的全量数据
def words_combiner_fuzzy(merged_results,pre_types=["SYM","ORG"], centertype="SUR"):
# po_series=set(po_series)
    po_series = []
    for pt in pre_types:
        po_series.append([pt,centertype])
        for t in all_tags:
            if t in exl_tags:
                continue
            po_series.append([pt,t,centertype])
    indications_copy = copy.deepcopy(merged_results)
    newly_merged = defaultdict(list)
    for j in indications_copy:
        j_dict = {}
        for z1,z2 in zip(list(range(0,len(j["seg"]))),j["seg"]):
            j_dict[z1] = z2
        try:
            segs = [s[1] for s in j["seg"]]
            words = [s[0] for s in j["seg"]]
        except:
            print("fuzzy error1", j["seg"])
        for seq in po_series:
            # iterating through all candidate sequences
            seqkey = "|".join(seq)
    #         newly_merged[seqkey].append("h")
            (pos,contains) = is_a_in_x(seq, segs)
            if contains: 
#                 try:
                start = j["seg"][pos-1][2][1]
                combined = "".join(words[pos:pos + len(seq)])
        
                #限制长度
                if len(combined) >= 12:
                    continue
                if "、" in combined or combined not in j["sentence"] or "," in combined or ":" in combined:
                    continue

                newly_merged[seqkey].append(combined)
                j_dict[pos] = [combined, seq[-1], [start, start + len(combined)]]
                print(combined, seq)
                try:
                    del j_dict[pos + 1]
                except:
                    continue
                if len(seq) == 3:
                    try:
                        del j_dict[pos + 2]
                    except:
                        continue
#                 except:
#                     print("fuzzy 2 error", j["seg"])
#         j_dic_newind = {}
#         for dic_ind, (k,v) in enumerate(j_dict.items()): 
#             j_dic_newind[dic_ind] = v
        j["seg"] = list(j_dict.values())
    lennew = 0
    for j in newly_merged.values():
        lennew += len(j)
    print("newly discovered combinations:", lennew) 
    return(indications_copy)
#         newly_merged.append(new_words)    
def round_1(merged_results):
    pre_copy1 = words_combiner_fuzzy(merged_results,pre_types = ["SYM","ORG","DIS","BFL","DEG","OGN"], centertype = "DIS")
    pre_copy1 = words_combiner_fuzzy(pre_copy1,pre_types = ["SYM","ORG","DIS","BFL","DEG","OGN"], centertype = "SYM")
    pre_copy1 = words_combiner_fuzzy(pre_copy1,pre_types = ["SYM","ORG","DIS","OBJ","DRU"], centertype="SUR")
    pre_copy2 = words_combiner_fuzzy(pre_copy1, pre_types = ["ORG","DIS","DIS","BFL"], centertype="TES")
    pre_copy3 = words_combiner_fuzzy(pre_copy2, pre_types = ["ORG", "DRU","OBJ"], centertype="EQM")
    pre_copy1 = words_combiner(pre_copy1, [["SUR", "DRU"], ["ORG", "SUR"], ["DIS", "SUR"], ["TES", "SUR"]])
    pre_copy1 = words_combiner(pre_copy1, [ ["SUR", "SUR"], ["ORG", "SUR"], ["DIS", "SUR"], ["TES", "SUR"]])
    return(pre_copy1)


def round_2(pre_copy1):
    symsuffix = [k for k,v in suffix_dic.items() if v == 'SYM']
    dissuffix = [k for k,v in suffix_dic.items() if v == 'DIS']
    eqsuffix = [k for k,v in suffix_dic.items() if v == 'EQM']
    tssuffix = [k for k,v in suffix_dic.items() if v == 'TES']
    srsuffix = [k for k,v in suffix_dic.items() if v == 'SUR']
    pre_copy1 = suffix_combiner(pre_copy1, symsuffix, desiredlist=["DIS","SYM","BFL","ORG"], suffixtype="SYM")
    pre_copy1 = suffix_combiner(pre_copy1, dissuffix, desiredlist=["DIS","SYM","BFL","ORG"], suffixtype="DIS")
    pre_copy1 = suffix_combiner(pre_copy1, eqsuffix, desiredlist=["SUR","TES","BFL","EQM"], suffixtype="EQM")
    pre_copy1 = suffix_combiner(pre_copy1, tssuffix,desiredlist=["SUR","TES","ORG","DIS","SYM","BFL"],suffixtype="TES")
    pre_copy1 = suffix_combiner(pre_copy1, srsuffix,desiredlist=["SUR","TES","ORG","DIS","SYM","BFL"],suffixtype="SUR")
    pre_copy1 = words_combiner(pre_copy1,[["SUR","SUR"],["ORG","SUR"],["DIS","SUR"]])
    pre_copy1 = words_combiner_fuzzy(pre_copy1,pre_types=["ORG","DIS","DRU","OBJ","BFL"], centertype="TES")
    pre_copy1 = words_combiner_fuzzy(pre_copy1,pre_types=["SYM","ORG","DIS","OBJ","DRU"], centertype="SUR")
    return(pre_copy1)


## 合并后缀
def suffix_combiner(merged_results, suffix, desiredlist, suffixtype):
# po_series=set(po_series)
    indications_copy=copy.deepcopy(merged_results)
    newly_merged=defaultdict(list)
    for j in indications_copy:
        j_dict={}
        
        for z1,z2 in zip(list(range(0, len(j["seg"]))), j["seg"]):
            j_dict[z1] = z2
        segs = [s[1] for s in j["seg"]]
        words = [s[0] for s in j["seg"]]
        for dt,md in enumerate(j["seg"][:-1]):
            if md[1] in desiredlist and j["seg"][dt+1][0] in suffix:
                pos = dt
                start = j["seg"][pos-1][2][1]
                combined = "".join(words[pos:pos+2])
                if "、" in combined or combined not in j["sentence"] or "," in combined or ":" in combined:
                    continue
                seqkey = md[1]
                newly_merged[seqkey].append(combined)
                j_dict[pos] = [combined, suffixtype, [start, start + len(combined)]]
                print(combined)
                del j_dict[pos + 1]

        j["seg"] = list(j_dict.values())
#     print(j["seg"])
    lennew = 0
    for j in newly_merged.values():
        lennew += len(j)
    print("newly discovered combinations:",lennew) 
    return(indications_copy)




class prefix_tool():
    def __init__(self, wordlist):
        print("length of list:",len(wordlist))
        self.start = time.time()
        self.dictionary_words_cut = []
        self.dictionary_flags_cut = []
        print("step 0")
        self.__init_run__(wordlist)
        print("step 1")
        self.__prefix_finder__()
        print("step 2")
    def __init_run__(self,wordlist):
        for ind,j in enumerate(wordlist):
            word_cut = []
            type_cut = []
            seg_list = jieba.posseg.cut(j)
#             indicator=False
            for s in seg_list:
                word_cut.append(s.word)
                type_cut.append(s.flag)
            self.dictionary_words_cut.append(word_cut)
            self.dictionary_flags_cut.append(type_cut)
            self.suffix = [(i[-1]+"|"+j[-1]) for i,j in zip(self.dictionary_words_cut,self.dictionary_flags_cut)]
            self.prefix = [(i[0]+"|"+j[0]) for i,j in zip(self.dictionary_words_cut,self.dictionary_flags_cut)]
            if ind == int(len(wordlist)/20):
                newtime = time.time()
                duration = newtime - self.start
                print("5 percent completed, took about %s mins, estimated time left %s"%(duration/60,duration/3))
            elif ind == int(len(wordlist)/2):
                newtime = time.time()
                duration = newtime-self.start
                print("50 percent completed, took about %s mins, estimated time left %s"%(duration/60,duration/60))

    def __prefix_finder__(self):
        suffixdic=defaultdict(int)
        prefixdic=defaultdict(int)
        for i in self.suffix:
            suffixdic[i] += 1
        for i in self.prefix:
            prefixdic[i] += 1
        self.sorted_prefix = {k: v for k, v in sorted(prefixdic.items(), key=lambda item: item[1])[::-1]}
        self.sorted_suffix = {k: v for k, v in sorted(suffixdic.items(), key=lambda item: item[1])[::-1]}

    def prefix_inspection(self,name,prefix=True):
        sameprefix = []
        if prefix == True:
            for i in self.dictionary_words_cut:
                if i[0] == name:
                    sameprefix.append("".join(i))
        if prefix == False:
            for i in self.dictionary_words_cut:
                if i[-1] == name:
                    sameprefix.append("".join(i))     
        return(sameprefix)
            

# plt.rcParams['figure.figsize'] = [10, 6]
 
def head_process(char, after):
    new = []
    uniques = []
    for [e, i, s, f] in after:
        e = str(e)
        if len(e) == 0:
            new.append([e, i, s, f])
            continue
        if e[0] == char or e[-1] == char:
            # unique.append(e)
            if e not in uniques:
                print(e)
                uniques.append(e)

            if e[0] == char:
                e = e[1:]
                s = s + 1
            elif e[-1] == char:
                e = e[:-1]
                f = f - 1
            if len(e) >= 2:
                new.append([e, i, s, f])
            else:
                new.append(["", i, s, f])
        else:
#             before.append([e, i, s, f])
            new.append([e, i, s, f])
    return(new)
               

def count_none(dic):
    counter = 0
    for k,v in dic.items():
        if v == None:
            counter += 1
    print(counter)
def main():
    global suffix_dic, exclusions_dic, nonessential_dic, essential_dic, all_tags, exl_tags, jieba_inverted
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path =  'logs/saving_'
    logfile = log_path + rq + '.log'
    logger.add(logfile, backtrace=True, diagnose=True, rotation='3 days', retention='2 months')
    with open('config.yaml', 'rb') as fp:
        dic_file = yaml.safe_load(fp)
        ont_map = dic_file['ontologies']['ontology_mapping']
        exclusions_dic = dic_file['dictionaries']['exclusions']
        suffix_dic = dic_file['dictionaries']['suffix']
        model_path = dic_file['model_path']
        desirable_models = dic_file['prediction_model']
        ner_training_reformatted = dic_file['training_data']['ner_training_reformatted']
        ner_training_final = dic_file['training_data']['ner_training_final']
        evaluation_path = dic_file['evaluation_path']
        ner_results = dic_file['results']['ner_results']

    ## clickhouse client related params and dics
        jieba_inverted = dic_file['ontologies']['jieba_inverted']
        all_tags = list(dic_file['ontologies']['jieba_inverted'].keys()) + ['x']
        essential_dic = dic_file['ontologies']['RE']
        types = list(essential_dic.keys())
        nonessential_dic = dic_file['ontologies']['RE_nonessential']
        tag_mapping = {**essential_dic, **nonessential_dic}
        full_dict = dic_file['dictionaries']['full']
        refined_dict = dic_file['dictionaries']['refined']
        full_jieba = dic_file['dictionaries']['full_jieba']
        ner_training_cleaned = dic_file['training_data']['ner_training_cleaned']
        clinicals = dic_file['corpus']['clinicals']
        manuals = dic_file['corpus']['manuals']
        version_no = dic_file['model_version']
    exclusions = []
    with open(exclusions_dic, "r") as f:
        for line in f:
            exclusions.append(line.strip())

    with open(suffix_dic) as f:
        suffix_dic = json.load(f)
    params= {"host": "192.168.4.30",
                "port": "9004",
                "user": "inside_a",
                "password": "KdtqKtJk",
                "database": "kg_cdm"}
    client = Client(host=params['host'],
                    user=params['user'],
                    port=params['port'],
                    password=params['password'],
                    database=params['database'], settings={'use_numpy': True})
    # prefix dic initializting
    exl_tags = ["u","m","x","c","p","r","d","v"]  

    extraction_result_file_new = "{}/extraction_results_{}_new.csv".format(ner_results, version_no)


    """
    2. 字典抽取结果的存储
    """
    logger.info("STEP 2. 字典抽取结果的存储")
    ## 头实体词典 Sep_m_dic, d2 d3 都是refined字典
    with open(refined_dict, "r") as rd:
        refined = json.load(rd)

    jieba.load_userdict(full_jieba)
    reverse_mapping = {v:k for (k,v) in jieba_inverted.items()}

    ## 对上一步储存得到的每一个数据集，分别进行字典匹配查找（all字典，jieba）
    for file_abb in ["m", "c", "d", "kd"]:
        print(file_abb)
        dic_entities = []
        with open("{}/{}_all_{}.json".format(ner_results, file_abb, version_no), "r") as evald:
            all3 = json.load(evald)
        for result in all3:
            sent = result["sentence"]
            if len(sent) <= 3:
                continue
            seg_list = jieba.posseg.cut(sent)
            word_cut = []
            p0 = 0
            p1 = 0
            for sind, s in enumerate(seg_list):
                p1 += len(s.word)
                position = [p0, p1]
                p0 = p1
                cutted_type = reverse_mapping.get(s.flag, "x")
                if cutted_type != "x":
                    ent_info = {}
                    if s.word in refined.keys():
                        prob = 0.95
                        model = "refined_dictionary"
 
                    else:
                        prob = 0.9
                        model = "other_dictionary"
                    if cutted_type == "SRS":
                        cutted_type = "SUR"
                    ent_info["ind"] = result["ind"]
                    ent_info["model"] = model                
                    ent_info["ent_name"] = s.word
                    ent_info["ent_type"] = cutted_type
                    ent_info["start"] = position[0]
                    ent_info["end"] = position[1]
                    try:
                        ent_info["ent_cn"] = tag_mapping[cutted_type]
                    except:
                        continue
                    ent_info["prob"] = prob
                    dic_entities.append(ent_info)
                word_cut.append([s.word, cutted_type, position])
            result["seg"] = word_cut
        dic_entities = pd.DataFrame(dic_entities)
        dic_entities.to_csv("{}/extraction_results_{}_dic_{}.csv".format(ner_results, version_no, file_abb))
        with open("{}/{}_all_{}_new.json".format(ner_results, file_abb, version_no), "w") as evald:
            json.dump(all3, evald)      
        
    """
    3， 对NER模型抽取结果进行整理, 主要是边界的标点的移除, 生成source信息表

    """
    logger.info("STEP 3， 对NER模型抽取结果进行整理, 主要是边界的标点的移除, 生成source信息表")
    dfs = []
    for csvfile in os.listdir("{}".format(ner_results)):
        if csvfile.endswith("_all.csv") and csvfile.startswith("extraction_"):
            dfs.append(pd.read_csv("{}/{}".format(ner_results,csvfile)))
    ner_result_all = pd.concat(dfs)

    ner_result = ner_result_all[ner_result_all.model == "ensemble_strong"]
    ner_result_list = ner_result.values.tolist()
    all_indexes = [i[-2] for i in ner_result_list]
    ents = list(ner_result.ent_name)
    inds = list(ner_result.ind)
    starts = list(ner_result.start)
    ends = list(ner_result.end)
    before = []
    for (e, i, s, f) in zip(ents, inds, starts, ends):
        before.append([e, i, s, f])

    after = copy.deepcopy(before)
    for symbol in list(",，：:、【】[]-.。&*！="):
        after = head_process(symbol, after)
    for symbol in list(",，：:、【】[]-.。&*！="):
        after = head_process(symbol, after)
    for symbol in list(",，：:、【】[]-.。&*！="):
        after = head_process(symbol, after)
    for symbol in list("-"):
        after = head_process(symbol, after)
    for symbol in list("-"):
        after = head_process(symbol, after)

    ## 修改边界，并且融入到ner结果表的dataframe中（只修改ensemble模型的结果）
    for i, l in enumerate(after):
        ner_result_list[i][2] = after[i][0]
        ner_result_list[i][3] = after[i][2]
        ner_result_list[i][4] = after[i][3]
    ner_result_new = pd.DataFrame(ner_result_list, columns = ner_result.columns)
    others = ner_result_all[ner_result_all.model != "ensemble_strong"]

    ner_result_full_new = pd.concat([ner_result_new, others])
    newcol = list(ner_result_full_new.columns)
    newcol[0] = "entity_ind"
    ner_result_full_new.columns = newcol

    ner_result_full_new.to_csv(extraction_result_file_new)

    # 获取index - sentence pair, 语料信息表构建并存储
    ## 获取index - sentence pair, 语料信息表构建并存储

    all_sent = []
    for rep in ['m','d','c','kd']:
        # "{}/{}_all_{}_new.json".format(ner_results, file_abb, version_no)
        with open("{}/{}_all_{}_new.json".format(ner_results, rep, version_no), "r") as evald:
            original = json.load(evald)
            for o in original:
                ind_sent = {}
                ind_sent["ind"] = o["ind"]
                ind_sent["sentence"] = o["sentence"]
                ind_sent["book"] = o.get("book")
                ind_sent["location"] = o.get("location")
                ind_sent["paragraph"] = o.get("paragraph")
                ind_sent["corpus"] = o["source"]  
                ind_sent["head_entity"] = o["entity1"] 
                ind_sent["head_type"] = o.get("entity1_type", "")
                all_sent.append(ind_sent)

    all_sent = pd.DataFrame(all_sent)
    all_sent = all_sent.sort_values("ind")
    clean_types = []
    for ele in all_sent.head_type:
        if type(ele) == list:
            ele = ele[0]
        clean_types.append(ele)
    all_sent.head_type = clean_types
    all_sent.to_csv("{}/source_info.csv".format(ner_results))
    ## 获取sentence信息
    ind_sent = {k:v for k,v in zip(list(all_sent.ind), list(all_sent.sentence))}
    sentences = []
    for x in ner_result_full_new.ind:
        sentences.append(ind_sent[x])
    ner_result_full_new["sentence"] = sentences
    ner_result_full_new = ner_result_full_new.sort_values("ind")
    ner_result_full_new.to_csv(extraction_result_file_new, encoding = "utf-8-sig")


    """
    # 4. NER merge with dictionaries
    # 5. merged according to rules, then save files

    # """
    logger.info("STEP 4 & 5. merged according to rules, then save files")

    for fileid in ['kd', 'm', 'c', 'd']:
        with open("{}/{}_all_{}_new.json".format(ner_results, fileid, version_no)) as f:
            indications = json.load(f)
        merged_results = merge_nerseg(indications)
        logger.info("STEP 4 & 5. merging results from : {}, round 1".format(fileid))
        pre_copy1 = round_1(merged_results)
        logger.info("STEP 4 & 5. merging results from : {}, round 2".format(fileid))
        pre_copy2 = round_2(pre_copy1)
        with open("{}/{}_final.json".format(ner_results, fileid), "w") as f:
            json.dump(pre_copy2, f)

    """
    6. data post-processing
    6.1. probability assignment
    """
    logger.info("STEP 6.1 probability assignment")

    ensemble = pd.read_csv(extraction_result_file_new)
    ensemble = ensemble[ensemble.model == "ensemble_strong"]

    """
    对句子的序列全部词赋予概率, 以字典匹配和ner匹配的概率最大的为准，
    字跨服内全部字都赋予这个概率并选两种方式的最大值
    随后查找合并后实体，计算区间内所有字符概率的均值
    """

    combined_ents = []
    for fileid in ['m','d','c','kd']:
        with open("{}/{}_final.json".format(ner_results, fileid)) as f:
            m_final = json.load(f)
    #     with open("data/ner_results/m_all_4.json") as f:
    #         m_original = json.load(f)
        m_dic = pd.read_csv("{}/extraction_results_{}_dic_{}.csv".format(ner_results, version_no, fileid))
        lentotal = len(m_final)
        current = 0
        for x in m_final:
            if current % 10000 == 0:
                print("{} | {}  / {}, current words accumulated: {}".format(fileid, current, lentotal, len(combined_ents)))
            current += 1
            index = x["ind"]
            ## 对句子的序列全部词赋予概率
            m_dic_sub = m_dic[m_dic.ind == index]
            ensemble_sub = ensemble[ensemble.ind == index]
            # print(x, "\n", ensemble_sub)
            prob_seq = [0] * len(x["sentence"])
            for st, ed, prob in zip(ensemble_sub.start, ensemble_sub.end, ensemble_sub.prob_strict):
                for i in range(st, ed):
                    prob_seq[i] = prob
            for st, ed, prob in zip(m_dic_sub.start, m_dic_sub.end, m_dic_sub.prob):
                for i in range(st, ed):
                    prob_seq[i] = np.max([prob_seq[i], prob])
            previous_entities = set(list(m_dic_sub.ent_name) + list(ensemble_sub.ent_name))
            # print("success")
            seg = x["seg"]
            # print(list(m_dic_sub.ent_name), list(ensemble_sub.ent_name), seg)
            maxseg = np.max([i[-1][-1] for i in seg])
            if maxseg > len(x["sentence"]):
                begins = 0
                for wi, w_list in enumerate(seg): 
                    ends = begins + len(w_list[0])
                    x["seg"][wi][2] = [begins, ends]
                    begins = ends
    #                 print(x["seg"])
            maxseg = np.max([i[-1][-1] for i in seg])
            if maxseg > len(x["sentence"]):
                continue
            for w_list in seg:
        #         print(w_list)
                if w_list[0] not in previous_entities and w_list[1] != "x":
                    combined_ent_dic = {}
                    probs_sum = 0
        #             print("hey", w_list[2][0], w_list[2][1])
                    for span in range(w_list[2][0], w_list[2][1]):
        #                 print(prob_seq[span], span)
        
                        probs_sum += prob_seq[span]
                    combined_ent_dic["model"] = "rules_merging"
                    combined_ent_dic["ent_name"] = w_list[0]   
                    combined_ent_dic["start"] = w_list[2][0]
                    combined_ent_dic["end"] = w_list[2][1]   
                    combined_ent_dic["prob"] = np.min([probs_sum/(span + 1) + 0.5, 0.98])
                    combined_ent_dic["ent_type"] = w_list[1]
                    combined_ent_dic["ind"] = x["ind"]     
                    combined_ents.append(combined_ent_dic)
            
    combined_ents_df = pd.DataFrame(combined_ents)
    combined_ents_df.to_csv("{}/combined_by_rules.csv".format(ner_results))
    suf = "&和的,」)(1后中前未不-且有性由人:："
    pre = "&和的,」)(未其均-已"
    mid = ",】【><：:。"
    exls_ent = []
    for cedid, x in zip(combined_ents_df.ind, combined_ents_df.ent_name):
        x = str(x)
        if x[-1] in suf or x[0] in pre:
            print(x)
            exls_ent.append(cedid)
        for m in mid:
            if m in x:
                exls_ent.append(cedid)            
    print(len(exls_ent))
    combined_ents_df = combined_ents_df[~combined_ents_df.ind.isin(exls_ent)]
    combined_ents_df = combined_ents_df.iloc[:, 1:]
    combined_ents_df.to_csv("{}/combined_by_rules.csv".format(ner_results))

    """
    6.2 所有结果合并，模型信息表/模型评估表存储
    """

    logger.info("STEP 6.2 所有结果合并，模型信息表/模型评估表存储")

    with open("{}/eval_dic.json".format(ner_results)) as jsoneval:
        evaluation_metrics = json.load(jsoneval)
    model_name = ['d', 'c', 'kd', 'm', 'a', 'ensemble_strong']
    model_parent = ["CNN_LSTM_CRF"]*5 + ["Ensemble"]
    training_set = ["药品说明书", "临床指南", "资料库", "操作手册", "全量数据",  "集成模型"]
    child_modes = [""]*5 + [",".join(model_name[:5])]
    code = ["kg/ner/"]*6
    config = [""]*6
    design = [""]*6
    model_info = pd.DataFrame({"model_name": model_name, "model_parent": model_parent, "training_set": training_set,
                            "child_modes": child_modes, "code_path": code, "config_path": config, "design_path": design})

    modelpaths = "{}/params_o_cnn_{}_params.pkl"
    modelabbre = ["d","c","kd","m","a"]
    model_info["model_path"] = [modelpaths.format(model_path, i) for i in modelabbre] + [""]

    model_info["creation_time"] = dt_string
    model_info["created_by"] = "朱佳"
    model_info.to_csv("{}/model_info.csv".format(ner_results))

    eval_table = []
    for dataset, level_1 in evaluation_metrics.items():
        for model, level_2 in level_1.items():
            for entity, probs in level_2.items():
                eval_table.append([model, dataset, entity] + probs + [probs[0], dt_string, "ZJ"])
    eval_table = pd.DataFrame(eval_table)
    eval_table.columns = ["model_name", "dataset_name", "entity", "strict_prob", "partial_prob", "exact_prob", "type_prob", "prob", "creation_date", "created_by"]
    eval_table.to_csv("{}/model_eval_results.csv".format(ner_results))
    """
    6.3 训练集/测试集信息存储

    """
    logger.info("6.3 训练集/测试集信息存储")
    training_set = ["药品说明书", "临床指南", "资料库",  "操作手册", "全量数据", "药品说明书",\
                    "临床指南", "资料库",  "操作手册", "全量数据", "药品说明书", "临床指南", "资料库",  "操作手册"] +[ "全量数据"] *5
    dataset_names = ["d", "c", "kd", "m", "a", "augmented_d", "augmented_c", "augmented_k",\
                    "augmented_m", "augmented_a"]
    dataset_paths = []
    for dcode in dataset_names: 
        path = "{}/{}_training.txt".format(ner_training_final, dcode)
        dataset_paths.append(path)
    for dcode in ["d", "c", "kd", "m"]: 
        path = "{}/{}_test.txt".format(ner_training_final, dcode)
        dataset_paths.append(path)
    dataset_names = [i.split("/")[-1] for i in dataset_paths]

    ## 数据随即替换版本的验证数据
    aug_test_names = ["seed", "batch1", "batch2", "batch3", "batch1_b"]
    aug_test_names = ["oob_test_{}.txt".format(i) for i in aug_test_names]
    aug_dataset_paths = ["{}/{}".format(ner_training_final, i) for i in aug_test_names]
    dataset_paths = dataset_paths + aug_dataset_paths
    dataset_names = dataset_names + aug_test_names

    dataset_names
    tagging_method = ["NER and dictionary matching"]*5 + ["NER and dictionary matching + data augmentation"]*5 +\
    ["NER and dictionary matching"]*4 +\
    ["NER and dictionary matching, with only unseen entities", 
    "NER and dictionary matching, with only unseen entities and entity random replacement strategy",
    "NER and dictionary matching, with only unseen entities and context noise injection", 
    "NER and dictionary matching, with only unseen entities, \
    entity random replacement from dictionary and context noise injection",
    "NER and dictionary matching, with only unseen entities and entity random replacement strategy"]
    generation_method = ["train test split： training "]*10 + ["train test split： testing "]*9
    creation_time = ["2022-6-30"]*19
    created_by = ["ZJ"] * 19
    usage = ["训练"] *10 + ["测试"] *9
    training_sets_info = pd.DataFrame({"corpus": training_set, "dataset_name": dataset_names, "dataset_paths": dataset_paths,
                            "tagging_method": tagging_method, "generation_method": generation_method, "usage":usage, "creation_time": creation_time, \
                            "created_by": created_by})

    training_sets_info["creation_time"] = dt_string
    training_sets_info["created_by"] = "ZJ"

    training_sets_info.to_csv("{}/training_info.csv".format(ner_results))

    """
    6.4 知识源信息存储

    """
    logger.info("6.4 知识源信息存储")

    cols = ["data_paths", "source_id", "source_name", "parent_id", "generation_method", "format", "script_path"]
    ## ocr 结果
    all_sources = []
    manuals_books = os.listdir(manuals)
    manuals_books = [i for i in manuals_books if ("txt" in i and "test" not in i)]
    for i, cm in enumerate(manuals_books):
        all_sources.append(["{}/{}".format(manuals, cm), "TXTC{0:0=2d}".format(i), cm.split("/")[-1].split(".")[0], "", "OCR", "txt", "kg/structurize_files.py"])
    clinicals_books = os.listdir(clinicals)
    clinicals_books = [i for i in clinicals_books if "txt" in i]
    for i, cm in enumerate(clinicals_books):
        all_sources.append(["{}/{}".format(clinicals, cm), "TXTM{0:0=2d}".format(i), cm.split("/")[-1].split(".")[0], "", "OCR", "txt", "../../kg/structurize_files.py"])
    # txt格式切分转化为excel
    pre_csv = {"../../kg/input_data/clinical_col_1.csv" : "临床指南",
    "../../kg/input_data/manual_May_modified_new.xlsx": "操作手册",
    "../../kg/input_data/drug_all.csv": "药品说明书",
    "../../kg/input_data/surgeries.csv": "资料库：手术",
    "../../kg/input_data/labs.csv": "资料库：检验",
    "../../kg/input_data/examinations.csv": "资料库：检查",
    "../../kg/input_data/diseases.csv": "资料库：疾病"}

    for i, (pc, name) in enumerate(pre_csv.items()):
        parent = ""
        method = "规则"
        if name == "临床指南": 
            parent = "TXTC"
        elif name == "操作手册": 
            parent = "TXTM"
        else: method = "初始"
        all_sources.append([pc, "CSV{0:0=2d}".format(i), name, parent, method, pc.split(".")[-1], "../../kg/books_preprocessing.ipynb"])

    all_sources_df = pd.DataFrame(all_sources)
    ## csv转化为单句信息的json
    pre_csv = {"c" : "临床指南",
    "m": "操作手册",
    "d": "药品说明书",
    "ki": "资料库",
    "kd": "资料库"}
    splitted = os.listdir("{}".format(ner_training_cleaned))
    for i, f in enumerate(splitted):
        path = "{}/{}".format(ner_training_cleaned, f)
        source_id = "JSN{0:0=2d}".format(i)
        parent_corpus = f.split("_")[0] 
        parag = f.split("_")[1].split(".")[0]
        source_name = pre_csv[parent_corpus] + " {}".format(parag)
        if parent_corpus == "c":
            parent_id = "CSV00"
        elif parent_corpus == "m":
            parent_id = "CSV01"
        elif parent_corpus == "kd":
            parent_id = "CSV02"
        elif parent_corpus == "labs":
            parent_id = "CSV04"
        elif parent_corpus == "surgeries":
            parent_id = "CSV03"
        elif parent_corpus == "examinations":
            parent_id = "CSV05"
        elif parent_corpus == "kd":
            parent_id = "CSV06"
        script = "kg/pre_labeling.py"
        all_sources.append([path, source_id, source_name, parent_id, "分句", "json", script])

    all_sources_df = pd.DataFrame(all_sources)
    all_sources_df.columns = ["data_paths", "source_id", "source_name", "parent_id", "generation_method", "format", "script_path"]



    all_sources_df["creation_time"] = dt_string
    all_sources_df["created_by"] = "ZJ"
    all_sources_df.to_csv("{}/corpus_info.csv".format(ner_results))
    logger.info(" 7. 临床路径结果整理（需要先单独跑出临床路径的预测结果，合并得到的结果，存储在下面读取的这个表中）")

    """
    7. 临床路径结果整理（需要先单独跑出临床路径的预测结果，合并得到的结果，存储在下面读取的这个表中）
    """

    clinical_path = pd.read_csv("/home/zhujia/kg/docextractor/final_results.csv")
    clinical_path.loc[clinical_path.TECH == 'RULE', 'prob'] = 0.97
    clinical_path.loc[clinical_path.TECH == 'FILL', 'prob'] = 0.96
    entity1 = [i.split("-")[0] for i in clinical_path.FILE]
    def mysplit(s):
        tail = s.lstrip('0123456789')
        head = s[:len(tail)]
        return head, tail
    
    entity1 = [mysplit(i)[1] for i in entity1]
    entity1 = [i.split("临床")[0] for i in entity1]
    entity1 = [i.split(".")[-1] for i in entity1]
    entity1 = [i.split("（")[0] for i in entity1]
    entity1_type = []
    for e1 in entity1:
        if "术" in e1:
            entity1_type.append("SUR")
        elif "疗" in e1 or "康复" in e1 or "取出" in e1:
            entity1_type.append("PRE")
        elif e1.endswith("镜"):
            entity1_type.append("TES")

        else:
            entity1_type.append("DIS")
    clinical_path["head_entity"] = entity1
    clinical_path["head_type"] = entity1_type
    clinical_path['corpus'] = 'p'
    print("cols", clinical_path.columns)
    clinical_corpus = clinical_path[["FILE", "IND", "YEAR", "SENTENCE", "head_entity", "head_type", "corpus"]].\
    drop_duplicates(["FILE", "IND", "SENTENCE", "head_entity", "head_type"])
    clinical_corpus.columns = ["book", "location", "year", "sentence", "head_entity", "head_type", "corpus"]
    pathid = [700000 + i for i in range(len(clinical_corpus))]
    clinical_corpus["ind"] = pathid
    clinical_corpus["paragraph"] = ""
    other_corpus = pd.read_csv("{}/source_info.csv".format(ner_results)).iloc[:, 1:]
    other_corpus = other_corpus[other_corpus.corpus != "p"]
    other_corpus["year"] = ""
    other_corpus = other_corpus[clinical_corpus.columns.tolist()]
    corpus_full = pd.concat([clinical_corpus, other_corpus])
    corpus_full2 = corpus_full.drop_duplicates(["sentence", "location", "book", "head_entity", "paragraph"], keep = 'first')
    corpus_full2.to_csv("{}/source_info_full.csv".format(ner_results))




    """
    7. 临床路径源文件整理

    7.1 语料源
    """
    logger.info("7. 临床路径源文件整理: 7.1 语料源")
    cps = pd.read_csv("{}/corpus_info.csv".format(ner_results))
    source_path = []
    source_id = []
    source_book = []
    rpath = "docextractor/data/docx-0917/"

    for f in os.listdir(rpath):
        idx = 0
        for ff in os.listdir(rpath + f):
            if "$" in ff: continue
            source_path.append(rpath + f + "/" + ff)
            source_id.append("P{}{}".format(f ,idx))
            source_book.append(ff)
            idx += 1
    path_corpus = pd.DataFrame({"data_paths": source_path, "source_id": source_id, "source_name": source_book})
    path_corpus["parent_id"] = ""
    path_corpus["generation_method"] = "original"
    path_corpus["format"] = "docx"
    path_corpus["script_path"] = ""

    path_corpus_full = pd.concat([path_corpus, cps])
    path_corpus_full.to_csv("{}/corpus_info.csv".format(ner_results))
    path_year_dic = {}
    for y in [2009, 2010, 2011, 2012, 2013, 2016, 2017, 2019]:
        path_year_dic[y] = {}
    for i,j in zip(list(path_corpus.source_id), list(path_corpus.source_name)):
        path_year_dic[int(i[1:5])][j] = i
        
    source_dic =  { j:i for (i, j) in zip(list(cps.source_id), list(cps.data_paths))}
    ## 赋予语料ID
    source_id = []
    for (s1, s2) in zip(list(corpus_full.book), list(corpus_full.corpus)):
        if s2 == "d":
            source_id.append("CSV02")
        elif s2 == "kd":
            source_id.append("CSV06")   
        elif s2 == "labs":
            source_id.append("CSV04")
        elif s2 == "surgeries":
            source_id.append("CSV03")  
        elif s2 == "examinations":
            source_id.append("CSV05")
        elif s2 == 'c':
    #         for k,v in source_dic.items():
    #             if v.startswith("TXTC"):
            s1_path = '{}/{}'.format(clinicals, s1)
            source_id.append(source_dic.get(s1_path,"")) 
        elif s2 == 'm':
    #         for k,v in source_dic.items():
    #             if v.startswith("TXTM"):
            s1_path = '{}/{}'.format(manuals, s1)
            source_id.append(source_dic.get(s1_path,"")) 
        elif s2 == 'p':
            bookyear = s1[-4:]
            bookname = s1[:-5] + ".docx"
            idss = path_year_dic[int(bookyear)][bookname]
            source_id.append(idss)
        else:
            source_id.append("")


    corpus_full["source_id"] = source_id
    corpus_full.to_csv("{}/source_info_full.csv".format(ner_results))

    """

    7.2 临床路径NER结果合并

    """
    logger.info(" 7.2 临床路径NER结果合并")
    clinical_path = clinical_path.drop_duplicates()
    clinical_joined = pd.merge(clinical_path, corpus_full2,  how='left', left_on = ['FILE', 'SENTENCE'],\
                            right_on = ['book', 'sentence'])
    clinical_joined = clinical_joined.drop_duplicates()

    reversed_tag_mapping = {v:k for (k,v) in tag_mapping.items()}
    reversed_tag_mapping['症状'] = 'SYM'
    reversed_tag_mapping['措施'] = 'SUR'
    reversed_tag_mapping['时长'] = 'DUR'
    reversed_tag_mapping['检验检查'] = 'TES'
    reversed_tag_mapping['手术'] = 'SUR'
    reversed_tag_mapping['耗材'] = 'EQM'
    starts = []
    ends = []
    tags = []
    for e, s, t in zip(list(clinical_joined.ENTITY), list(clinical_joined.SENTENCE), list(clinical_joined.TAG2)):
        try:
            starts.append(s.index(e))
            ends.append(s.index(e) + len(e))
        except:
            starts.append(-1)
            ends.append(-1)
        tags.append(reversed_tag_mapping[t])
    clinical_joined["start"] = starts
    clinical_joined["end"] = ends
    clinical_joined["ent_type"] = tags
    clinical_joined["prob_partial"] = starts
    clinical_joined["prob_exact"] = ends
    clinical_joined["prob_type"] = tags                
    clinical_joined = clinical_joined[['Unnamed: 0', 'TECH', 'ENTITY', 'start', 'end', 'ent_type', 'prob', \
                                        'prob_partial', 'prob_exact', 'prob_type', 'ind', 'TAG2', 'SENTENCE']]

                           
    ## index need to be fixed, also need to take whole results into consideration. 
    clinical_joined = clinical_joined.drop_duplicates().iloc[:,1:]
    clinical_joined.columns = ["model", "ent_name", "start", "end", "ent_type", "prob", 'prob_partial', 'prob_exact', 'prob_type', "ind", "ent_cn", "sentence"]   
    # clinical_joined.to_csv(extraction_result_file_new)

    """
    8. 全部NER结果整理，合并，概率变更

    """
    logger.info("STEP 8. 全部NER结果整理，合并，概率变更")
    ner_results_df = pd.read_csv(extraction_result_file_new).iloc[:,1:]
    ner_results_df.columns = ["entity_ind", "model", "ent_name", "start", "end", "ent_type", "prob", 'prob_partial', 'prob_exact', 'prob_type', "ind", "ent_cn"]  
    ner_results_df = pd.merge(ner_results_df, corpus_full2,  how='left', left_on = [ 'ind'],\
                            right_on = ['ind']) 
    ner_results_df = ner_results_df[["entity_ind", "model", "ent_name", "start", "end", "ent_type", "prob", 'prob_partial', 'prob_exact', 'prob_type', "ind", "ent_cn", "sentence"]]                  
    dic_results = []
    for df in os.listdir(ner_results):
        if "{}_dic".format(version_no) in df:
            df_data = pd.read_csv("{}/{}".format(ner_results, df))
            dic_results.append(df_data)
    dic_results = pd.concat(dic_results)
    dic_results.loc[dic_results.ent_name.str.len() <= 2, "prob"] = 0.7
    dic_results_cols = list(dic_results.columns)
    try:
        dic_results_cols.remove('model')
    except:
        pass
    dic_results = pd.merge(dic_results, corpus_full2, how = "left", left_on = ["ind"], right_on = ["ind"])
    rules = pd.read_csv("{}/combined_by_rules.csv".format(ner_results))
    rules = pd.merge(rules, corpus_full2, how = "left", left_on = ["ind"], right_on = ["ind"])
    rules["ent_cn"] = [tag_mapping[i] for i in list(rules.ent_type)]
    rules_results = rules[dic_results_cols[1:] + ["sentence"]]
    dic_results = dic_results[dic_results_cols + ["sentence"]].iloc[:,1:]
    dic_results["model"] = 'DICT'
    dic_results["entity_ind"] = 0

#     final_cols = list(dic_results.columns)
#     final_cols[7] = "prob_strict"
    dic_results.columns = ['ind', 'ent_name', 'ent_type', 'start', 'end', 'ent_cn','prob', 'sentence', 'model', 'entity_ind']
    rules_results['model'] = 'COMBINED_BY_RULES'
    rules_results['entity_ind'] = 0
    desired_cols = rules_results.columns
    rules_results = rules_results[desired_cols]
    clinical_joined['entity_ind'] = 0
    clinical_joined = clinical_joined[desired_cols]
    ner_results_df = ner_results_df[desired_cols]
    for dfx in [ner_results_df, clinical_joined, dic_results, rules_results]:
        print("df \n", dfx[dfx.model == '呼吸困难'],"\n")
    
    ## 非临床路径结果，临床路径结果， 字典匹配结果，规则合并结果
    all_entities = pd.concat([ner_results_df, clinical_joined, dic_results, rules_results])
    all_entities = all_entities.sort_values("ind")
    inds = [i for i in range(len(all_entities))]
    all_entities['entity_ind'] = inds
    all_entities_2 = all_entities.sort_values("prob", ascending = False).drop_duplicates(["ind", \
                                                                                                "ent_name", "ent_type", "start"], keep = 'first')
    all_entities_2 = all_entities_2.sort_values(["ind", "start"])
    all_entities_2.loc[all_entities_2.model == 'NER', 'model'] = 'a'
    all_entities_2.to_csv("{}/all_entities.csv".format(ner_results))

    """
    9. 最终清理，并写入数据库
    """
    logger.info("STEP 9. 最终清理，并写入数据库")
    type_dic = {}
    for en, cn in zip(list(all_entities_2.ent_type), list(all_entities_2.ent_cn)):
        type_dic[en] = cn
    corpus_info = pd.read_csv("{}/corpus_info.csv".format(ner_results)).iloc[:, 1:]
    corpus_info["source_type"] = ""
    corpus_info.loc[corpus_info.source_id.str.startswith("P") == True, "source_type"] = "临床路径"
    corpus_info.loc[corpus_info.source_id.str.startswith("JSN") == True, "source_type"] = "药品说明书"
    corpus_info.loc[corpus_info.source_id.str.startswith("CSV") == True, "source_type"] = "资料库"
    corpus_info.loc[corpus_info.source_id == 'CSV00', "source_type"] = "临床指南"
    corpus_info.loc[corpus_info.source_id == 'CSV01', "source_type"] = "操作手册"
    corpus_info.loc[corpus_info.source_id == 'CSV02', "source_type"] = "药品说明书"
    corpus_info.loc[corpus_info.source_id.str.startswith("TXTC") == True, "source_type"] = "临床指南"
    corpus_info.loc[corpus_info.source_id.str.startswith("TXTM") == True, "source_type"] = "操作手册"
    corpus_info.to_csv("{}/corpus_info.csv".format(ner_results))
    source_info = pd.read_csv("{}/source_info_full.csv".format(ner_results)).iloc[:, 1:]
    source_info["source_id"] = ""
    # ## 赋予语料ID
    source_id = []
    for (s1, s2) in zip(list(source_info.book), list(source_info.corpus)):
    #     print(s1, s2)
        if s2 == "d":
            source_id.append("CSV02")
        elif s2 == "kd":
            source_id.append("CSV06")   
        elif s2 == "labs":
            source_id.append("CSV04")
        elif s2 == "surgeries":
            source_id.append("CSV03")  
        elif s2 == "examinations":
            source_id.append("CSV05")
        elif s2 == 'c':
    #         for k,v in source_dic.items():
    #             if v.startswith("TXTC"):
            s1_path = '{}/{}'.format(clinicals, s1)
            source_id.append(source_dic.get(s1_path,"")) 
        elif s2 == 'm':
    #         for k,v in source_dic.items():
    #             if v.startswith("TXTM"):
            s1_path = '{}/{}'.format(manuals, s1)
            source_id.append(source_dic.get(s1_path,"")) 
        elif s2 == 'p':
            bookyear = s1[-4:]
            bookname = s1[:-5] + ".docx"
            idss = path_year_dic[int(bookyear)][bookname]
            source_id.append(idss)
        else:
            source_id.append("")
    source_info.to_csv("{}/source_info_full.csv".format(ner_results))
    source_info["source_id"]= source_id
    source_segment = source_info[["source_id", "ind", "location", "head_entity", "paragraph", "head_type", "sentence"]]
    source_segment["crte_time"] = dt_string
    source_segment["crter"] = '朱佳'
 
    onto_id = []
    for x in source_segment.head_type:
        onto_id.append(ont_map.get(x, ""))



    # source segment

    delimiter = ','
    def string_handling(value):
        return(str(value).replace("'","\\'").replace(")","\)").replace("(","\("))
    source_segment["head_type"] = onto_id
    source_segment.columns = ['souc_id',
    'text_id',
    'catalog',
    'text_label',
    'text_background',
    'head_onto_id',                      
    'text_content',
    'crte_time',
    'crter']
    if os.path.isdir("{}/final".format(ner_results)) is False:
        os.makdirs("{}/final".format(ner_results))
    source_segment.to_csv("{}/final/source_segment.csv".format(ner_results))
    # source segmentation 添加头实体ID
    source_segment = pd.read_csv("{}/final/source_segment.csv".format(ner_results)).iloc[:, 1:]
    source_segment["text_background"] = source_segment["text_background"].astype(str)
    source_segment["catalog"] = source_segment["catalog"].astype(str)
    source_segment["text_label"] = source_segment["text_label"].astype(str)
    source_segment["souc_id"] = source_segment["souc_id"].astype(str)
    source_package = source_segment.groupby(["text_label", "text_background", "catalog" ,"souc_id"])
    all_packages = []
    for i, gi in enumerate(list(source_package.groups.keys())):
        package = source_package.get_group(gi)
        package["head_id"] = 'h{:08d}'.format(i)
        all_packages.append(package)
    all_packages = pd.concat(all_packages)
    segs = all_packages.sort_values("text_id")
    segs.loc[(segs.text_background == 'clinical')&(segs.souc_id.isnull()), "souc_id"] = 'JSN00'
    segs.loc[(segs.text_background == 'treatment')&(segs.souc_id.isnull()), "souc_id"] = 'JSN12'
    segs.loc[(segs.text_background == 'diagnosis')&(segs.souc_id.isnull()), "souc_id"] = 'JSN11'
    segs.loc[(segs.text_background == 'contraindiction')&(segs.souc_id.isnull()), "souc_id"] = 'JSN17'
    segs.loc[(segs.text_background == 'indications')&(segs.souc_id.isnull()), "souc_id"] = 'JSN13'
    segs.loc[(segs.text_background == 'procedures')&(segs.souc_id.isnull()), "souc_id"] = 'JSN14'
    segs.to_csv("{}/final/source_segment_2.csv".format(ner_results))
    all_values = segs.to_dict('records')
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    # for ele in all_values:
    #     ele['crte_time'] = dt_string
    #     replacement = delimiter.join(["\'{}\'".format(string_handling(value)) for value in tuple(ele.values())])
    #     try:
    #         client.execute("INSERT INTO kg_cdm.source_segment VALUES ({})".format(replacement))
    #     except:
    #         replacement = delimiter.join(["\'{}\'".format(value) for value in tuple(ele.values())])
    #         client.execute("INSERT INTO kg_cdm.source_segment VALUES ({})".format(replacement))        



    # source info
    corpus_info = corpus_info[["source_id", "parent_id", "generation_method",\
                            "source_type", "source_name", "format", "script_path", "data_paths"]]
    corpus_info["crte_time"] = dt_string
    corpus_info["crter"] = '朱佳'
    corpus_info.columns = ['souc_id',
    'father_souc_id',
    'gene_meth',
    'souc_type',
    'souc_name',
    'souc_form',
    'code_save',
    'souc_save',
    'crte_time',
    'crter']
    corpus_info.to_csv("{}/final/source_info.csv".format(ner_results))
    source_info = pd.read_csv("{}/final/source_info.csv".format(ner_results)).iloc[:, 1:]
    # client.insert_dataframe('INSERT INTO source_info VALUES', source_info)


    # source labeled
    source_labeled = pd.read_csv("{}/training_info.csv".format(ner_results)).iloc[:, 1:]
    for ctype in ["药品说明书", '操作手册', '临床指南', '资料库']:
        list(corpus_info[(corpus_info.souc_form == 'json') & (corpus_info.souc_type == ctype)]['souc_id'])
    training_source = []
    for c in source_labeled.corpus:
        if c == '全量数据':
            training_source.append(list(corpus_info[(corpus_info.souc_form == 'json')]['souc_id']))
        else:
            training_source.append(list(corpus_info[(corpus_info.souc_form == 'json') \
                                                    & (corpus_info.souc_type == c)]['souc_id']))
    source_labeled["source_id"] = training_source
    source_labeled = source_labeled[["dataset_name", "source_id", "tagging_method", "generation_method", "dataset_paths"]]
    source_labeled["crte_time"] = dt_string
    source_labeled["crter"] = '朱佳'
    source_labeled.columns = ['dataset_id',
    'souc_id',
    'label_meth',
    'gene_meth',
    'dataset_save',
    'crte_time',
    'crter']
    source_labeled.to_csv("{}/final/source_labeled.csv".format(ner_results))
    source_labeled = pd.read_csv("{}/final/source_labeled.csv".format(ner_results)).iloc[:, 1:]
    # client.insert_dataframe('INSERT INTO source_labeled VALUES', source_labeled)


    # ner info
    ner_info = pd.read_csv("{}/model_info.csv".format(ner_results)).iloc[:, 1:]
    ner_info["model_mingcheng"] = ["药品说明书训练的CNN_LSTM_CRF", "临床指南训练的CNN_LSTM_CRF", "资料库训练的CNN_LSTM_CRF",\
                                "全量数据训练的CNN_LSTM_CRF", "操作手册训练的CNN_LSTM_CRF2",\
                                "集成模型"]
    ner_info["dataset_id"] = ["d_training.txt", "c_training.txt", "kd_training.txt", "a_training.txt", "m_training.txt", "a_training.txt"] 
    ner_info = ner_info[["model_name", "model_mingcheng", "child_modes", "dataset_id", "design_path", "model_path", "code_path"]]
    ner_info["crte_time"] = dt_string
    ner_info["crter"] = '朱佳'
    ner_info.columns = ['ner_model_id',
    'ner_model_name',
    'submodel_list',
    'dataset_id',
    'doc_save',
    'code_save',
    'conf_save',
    'crte_time',
    'crter']
    ner_info.to_csv("{}/final/ner_info.csv".format(ner_results))
    ner_info = ner_info.fillna("")
    # client.insert_dataframe('INSERT INTO ner_info VALUES', ner_info)


    # ner eval
    ner_eval = pd.read_csv("{}/model_eval_results.csv".format(ner_results)).iloc[:, 1:]
    evalcol = list(ner_eval.columns)
    evalcol = ['model_name',
    'dataset_name',
    'entity',
    'strict_prob',
    'exact_prob',
    'type_prob',
    'partial_prob',
    'prob',
    'creation_date',
    'created_by']
    ner_eval = ner_eval[evalcol]
    ner_eval.columns = ['ner_model_id',
                        'dataset_id',
                        'onto_id',
                        'conf_precision',
                        'conf_coordinate',
                        'conf_entitytype',
                        'conf_vague_coor',
                        'model_conf',
                        'crte_time',
                        'crter']
    ner_eval['crter'] = "朱佳"
    ner_eval.to_csv("{}/final/ner_eval.csv".format(ner_results))
    ner_eval = pd.read_csv("{}/final/ner_eval.csv".format(ner_results)).iloc[:, 1:]
    # client.insert_dataframe('INSERT INTO ner_eval VALUES', ner_eval)

    """
    ner result and post-model cleansing
    """
    #ner result
    all_entities_2 = all_entities_2[["entity_ind", "model", "ind", "sentence", "ent_type", "ent_name", "start", "end",\
                                    "prob"]]
    all_entities_2["crte_time"] = dt_string
    all_entities_2["crter"] = '朱佳'
    all_entities_2.columns = ['entity_id',
    'ner_model_id',
    'text_id',
    'text_content',
    'onto_id',
    'entity_name',
    'entity_begn_coord',
    'entity_end_coord',
    'entity_conf',
    'crte_time',
    'crter'] 

    ## 单字的降低置信度
    ner_result = all_entities_2
    mask1 = ner_result.entity_name.str.len() == 1
    mask2 = (~ner_result.onto_id.isin(['AT', 'ORG', 'DEG', 'BFL']))
    ner_result.loc[mask1 & mask2,'entity_conf'] = ner_result[mask1 & mask2]["entity_conf"]/4

    # 减少双字的置信度
    mask1 = ner_result.entity_name.str.len() == 2
    mask2 = (ner_result.onto_id.isin(['SYM','TES','DIS','SUR','EQM']))
    mask3 = ner_result.entity_conf >= 0.2
    ner_result.loc[mask1 & mask2 & mask3,'entity_conf'] = ner_result[mask1 & mask2 & mask3]["entity_conf"]/3

    ## 部分常见错误减低置信度
    mask1 = ner_result.entity_name.str.endswith("触及")
    mask2 = ner_result.entity_name.str.endswith("闻及")
    ner_result.loc[mask1 | mask2,'entity_conf'] = 0.01

    companioned = ['伴有', '合并', '常伴', '并发', '有时', '伴或', '多发', '多伴', '以及', '如']
    elimination = ['预防', '晚期', '术后', '成人', '影响', '其他', '由于', '有时', '有无', '但', '一般', '但无', '主要', '切除', '固定', '养血',\
    '术中', '过敏', '其他', '处理','遵守', '常', '其', '使用', '多于', '仅']

    ## 后续词汇降低概率
    ## 部分伴有词移除（不降低概率）
    ## 部分词过滤
    ## 其余的添加但是降低概率
    eid = np.max(ner_result.entity_id) + 1
    new_dic = []
    ner_result["entity_name"] = ner_result["entity_name"].astype(str)
    for splitter in list('，,；;:：。'):
        mask1 = ner_result.entity_name.str.contains(splitter, na=False)
        mask2 = ner_result.entity_name.str.contains("\（|\(|\[", na=False)
        mask3 = ner_result.entity_conf >= 0.2
        ner_result.loc[(mask1) & (~mask2) & (mask3), "entity_conf"]  = (ner_result[mask1 & (~mask2) & mask3]["entity_conf"])/4
        subdics = ner_result[(mask1) & (~mask2) & (mask3)].to_dict('records')
        print(len(subdics), len(new_dic))
        for dic in subdics:
            newdics = dic['entity_name'].split(splitter)
        #     print(dic['entity_conf'], entity_conf)
            entity_conf = 4*dic['entity_conf'] + (1-4*dic['entity_conf'])/2
            crte_time = "".join(str(datetime.now()).split(".")[:-1])
            head = dic['entity_begn_coord']
            tail = head - 1
            ner_model_id = dic['ner_model_id'] + '_cleaned'
            
            order = 0
            for nd in newdics:
                order += 1
                head = tail + len(splitter)
                tail = head + len(nd)
                ## candidate too short to be regarded as a valid entity
                if len(nd) <= 2:                 
                    continue          
                else:
                    if order >= 2:
                        if any(nd.startswith(c) for c in companioned):
                            len_splitword = len(companioned[[nd.startswith(c) for c in companioned].index(True)])
                            nd = nd[len_splitword:]
                            head = head + len_splitword
                            entity_conf = 4*dic['entity_conf']
                        elif any(nd.startswith(c) for c in elimination):
                            continue
                        else:
                            entity_conf = 3*dic['entity_conf']
    
                eid += 1
                dicnew = copy.deepcopy(dic)
                dicnew['entity_id'] = eid
                dicnew['entity_name'] = nd
                dicnew['entity_conf'] = entity_conf
                dicnew['entity_begn_coord'] = head
                dicnew['entity_end_coord'] = tail
                dicnew['crte_time'] = crte_time
                dicnew['ner_model_id'] = ner_model_id
                new_dic.append(dicnew)

    mask0 = ner_result.entity_conf >= 0.2
    mask1 = (ner_result.entity_name.str.endswith(']'))&(~ner_result.entity_name.str.contains('\['))
    mask2 = (ner_result.entity_name.str.endswith(')'))&(~ner_result.entity_name.str.contains('\('))
    mask2 = (ner_result.entity_name.str.endswith(')'))&(~ner_result.entity_name.str.contains('\('))
    maskn = ner_result['entity_name'].str.endswith(tuple('<【[(（'))
    mask3 = ner_result.entity_name.str.endswith(tuple('或和及等与'))
    ner_result.loc[(mask1|mask2|mask3|maskn) & mask0, 'entity_conf'] =\
        ner_result[(mask1|mask2|mask3|maskn) & mask0]['entity_conf']/3
    subdics = ner_result[(mask1|mask2|mask3|maskn) & mask0].to_dict('records')

    for dic in subdics:
        dicnew = copy.deepcopy(dic)
        nd = dic['entity_name'][:-1]
        if len(nd) <= 2: continue
    #     print(dic['entity_conf'], entity_conf)
        entity_conf = dic['entity_conf']
        crte_time = "".join(str(datetime.now()).split(".")[:-1])
        tail = dic['entity_end_coord']
        ner_model_id = dic['ner_model_id'] + '_cleaned'
        eid += 1
        dicnew['entity_id'] = eid
        dicnew['entity_name'] = nd
        dicnew['entity_conf'] = entity_conf + (1 - entity_conf)/2
        dicnew['entity_begn_coord'] = head
        dicnew['entity_end_coord'] = tail - 1
        dicnew['crte_time'] = crte_time
        dicnew['ner_model_id'] = ner_model_id
        new_dic.append(dicnew)
    mask3 = ner_result.entity_name.str.endswith(tuple('[(（【'))
    mask0 = ner_result.entity_conf >= 0.3
    subdics = ner_result[mask3 & mask0].to_dict('records')
    ner_result.loc[mask3 & mask0, 'entity_conf'] = ner_result[mask3 & mask0]['entity_conf']/2
    splitter_list = list("（([【")
    for dic in subdics:
        dicnew = copy.deepcopy(dic)
        nd = dic['entity_name']
        splitter = splitter_list[[c in nd for c in splitter_list].index(True)]
        nd = nd.split(splitter)[0]
        if len(nd) <= 2: continue
    #     print(dic['entity_conf'], entity_conf)
        entity_conf = dic['entity_conf']
        crte_time = "".join(str(datetime.now()).split(".")[:-1])
        tail = dic['entity_end_coord']
        ner_model_id = dic['ner_model_id'] + '_cleaned'
        eid += 1
        dicnew['entity_id'] = eid
        dicnew['entity_name'] = nd
        dicnew['entity_conf'] = entity_conf + (1 - entity_conf)/2
        dicnew['entity_begn_coord'] = head
        dicnew['entity_end_coord'] = head + len(nd)
        dicnew['crte_time'] = crte_time
        dicnew['ner_model_id'] = ner_model_id
        new_dic.append(dicnew)
    mask3 = ner_result.entity_name.str.endswith(tuple('性型'))
    mask0 = ner_result.entity_conf >= 0.3
    mask1 = ner_result.entity_name.str.endswith(('典型', '血型', '阳性', '变性'))
    newcol = ner_result[(mask3) & (mask0) & (~mask1)]['entity_conf']/2
    ner_result.loc[(mask3) & (mask0) & (~mask1), 'entity_conf'] = newcol

    # 
    mask3 = ner_result.entity_name.str.endswith(tuple('下中前上'))
    mask1 = ner_result.onto_id.isin(['DIS','SYM','SUR','EQM','TES'])
    mask2 = ner_result.entity_name.str.endswith(('卒中', '低下'))
    ner_result.loc[(mask3) & (mask1) & (~mask2), 'entity_conf'] = ner_result[(mask3) & (mask1) & (~mask2)]['entity_conf']/3

    mask3 = ner_result.entity_name.str.endswith('时')
    mask1 = ner_result.onto_id.isin(['DIS','SYM','SUR','EQM','TES'])
    mask2 = ner_result.entity_name.str.endswith(('卒中', '低下'))
    ner_result.loc[(mask3) & (mask1) & (~mask2), 'entity_conf'] = ner_result[(mask3) & (mask1) & (~mask2)]['entity_conf'] * 0.8

    ner_result['crte_time'] = dt_string
    new_dic_df = pd.DataFrame(new_dic)
    ner_result = ner_result.append(new_dic_df)
    ner_result.to_csv("{}/final/ner_result.csv".format(ner_results))
#     ner_result = ner_result[ner_result.onto_id != ""]
#     for x in ner_result.onto_id:
#         onto_id.append(ont_map.get(x, ""))
#     ner_result["onto_id"] = onto_id
#     all_values = ner_result.to_dict('records')
#     i = 0
    # for ele in all_values:
    #     i += 1
    # #     if i <= 3939536: continue
    #     if i%10000 == 0 : print(i)
    #     try:
    #         replacement = delimiter.join(["\'{}\'".format(string_handling(value)) for value in tuple(ele.values())])
    #         client.execute("INSERT INTO kg_cdm.ner_result VALUES ({})".format(replacement))
    #     except:
    #         print("error")

    # ## 头实体写入

    # head_entities = client.execute("""select head_id, 'head_entity', text_id, text_content, head_onto_id, text_label, -1, -1, 0.97 from kg_cdm.source_segment""")
    # delimiter = ','
    # # def string_handling(value):
    # #     return(str(value).replace("'","\\'").replace(")","\)").replace("(","\("))
    # i = 0
    # for ele in head_entities:
    #     ele = list(ele)
    #     ele.append(dt_string)
    #     ele.append('朱佳')
    #     i += 1
    # #     if i <= 3939536: continue
    #     if i%10000 == 0 : print(i)
    #     try:
    #         replacement = delimiter.join(["\'{}\'".format(string_handling(value)) for value in ele])
    #         client.execute("INSERT INTO kg_cdm.ner_result VALUES ({})".format(replacement))
    #     except:
    #         print("error")


if __name__ == "__main__":

    main()
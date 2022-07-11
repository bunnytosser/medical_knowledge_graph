"""
主要功能：pre_labeling.py 得到json文件后，对标注结果进行后续处理。后续处理内容包括：
        1. 将NER结果和字典匹配结果进行合并整理，并取较大边界作为结果。如： ner:['劳力性呼吸困难'], seg:['呼吸困难'] ====> seg ['劳力性呼吸困难']
        2. 利用规则对实体进行合并，如：人体组织结构 + 临床表现 = 临床表现，疾病 + 观测操作 = 观测操作等。例如 胸部(ORG)X光(观测操作)===>胸部X光（观测操作）。
        会进行数轮迭代。
        3. 后续清洗，某些不合适的实体去除
        4. 分别划分训练，测试，验证集并且存储。

使用场景：模型训练。用于训练NER模型前最后一步的训练数据生成

"""

import pandas as pd
import os
import re
import requests
import numpy as np
import pandas as pd
import ast
import json
import jieba.posseg
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import seaborn as sns
from utils import prefix_tool, Utillist, Model_selector, words_combiner, suffix_combiner
import yaml
import os
import copy
from loguru import logger
import time



def lengthcleaner(n):
    if len(n[0]) == int(n[2][1])-int(n[2][0]):
        pass

    elif len(n[0]) == int(n[2][1])+1-int(n[2][0]):
        n[2] = [n[2][0],n[2][1]+1]
    return(n)

def nerlencal(ner):
    newn = []
    for n in ner:
        if "、" in n[0]:
            n_tag=n[1]
            n_start=n[2][0]
            splittedwords = n[0].split("、")
            updated_position = n_start
            for i,w in enumerate(splittedwords):
                if i == 0: 
                    updated_position=n_start+len(w)
                    newn.append([w,n_tag,[n_start,updated_position]])
                else:
                    newn.append([w,n_tag,[updated_position,updated_position+len(w)]])
                    updated_position=updated_position+len(w)
                    
        else:      
            n = lengthcleaner(n)
            newn.append(n)     
    return(newn)

def get_BI_word(xele_o, f, mapping = True):
    xele = copy.deepcopy(xele_o)
    tags = xele["seg"]
    sentence = xele["sentence"]
    if mapping == False:
        type_list = [v for v in type_mapping.values()]
    else:
        type_list = [k for k in type_mapping.keys()]
    for w in tags:
        if w[1] not in type_list:
            if len(w[0]) == 1:
                f.write(w[0] + '\t' + 'O' + '\n')
            else:
                splitted = list(w[0])
                f.write(splitted[0] + '\t' + 'O' + '\n')
                for i in range(len(splitted)-1):
                    f.write(splitted[i+1] + '\t' + 'O' + '\n' )
        else:
            if mapping == True:
                if len(w[0]) == 1:
                    f.write(w[0] + '\t' + 'B-' + type_mapping[w[1]] + '\n')
                else:
                    splitted = list(w[0])
                    f.write(splitted[0] + '\t' + 'B-' + type_mapping[w[1]] + '\n')
                    for i in range(len(splitted)-1):
                        f.write(splitted[i+1] + '\t' + 'I-' + type_mapping[w[1]] + '\n')
            else:
                if len(w[0]) == 1:
                    f.write(w[0] + '\t' + 'B-' + w[1] + '\n')
                else:
                    splitted = list(w[0])
                    f.write(splitted[0] + '\t' + 'B-' + w[1] + '\n')
                    for i in range(len(splitted)-1):
                        f.write(splitted[i+1] + '\t' + 'I-' + w[1] + '\n')      
    f.write('\n')

if __name__ == "__main__":
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path =  'logs/generation'
    logfile = log_path + rq + '.log'
    logger.add(logfile, backtrace=True, diagnose=True, rotation='3 days', retention='2 months')
    logger.info("STAGE 0: Initializing ")
    with open('config.yaml', 'rb') as fp:
        dic_file = yaml.safe_load(fp)
        merging_mode = dic_file['setting']['merging'] ### 是否尽量详尽得合并实体
        dic_generation = dic_file['setting']['unseen_dic'] ### 是否尽量详尽得合并实体
        clinicals = dic_file['corpus']['clinicals']
        manuals = dic_file['corpus']['manuals']
        instructions = dic_file['corpus']['instructions']
        knowledge_base = dic_file['corpus']['knowledge_base']
        re_dic = dic_file['ontologies']['RE']
        stop_list = dic_file['dictionaries']['stop']
        full_dict = dic_file['dictionaries']['full']
        refined_dict = dic_file['dictionaries']['refined']
        unseen = dic_file['dictionaries']['unseen']
        exclusions_dic = dic_file['dictionaries']['exclusions']
        suffix_dic = dic_file['dictionaries']['suffix']

        preprocessed_clinical = dic_file['data_files']['preprocessed_clinical']
        preprocessed_manual = dic_file['data_files']['preprocessed_manual']
        ## 训练数据存储的文件夹
        ner_training = dic_file['training_data']['ner_training']
        ner_training_cleaned = dic_file['training_data']['ner_training_cleaned']
        ner_training_reformatted = dic_file['training_data']['ner_training_reformatted']
        ner_training_final = dic_file['training_data']['ner_training_final']
        full_jieba = dic_file['dictionaries']['full_jieba']
        #jieba mapping
        jieba_inverted = dic_file['ontologies']['jieba_inverted']


    type_mapping = {v:k for k,v in jieba_inverted.items()}
    exclusions = []

    with open(stop_list,"r") as f:
        for line in f:
            exclusions.append(line.strip())
    with open(exclusions_dic) as f:
        all_exl = f.readlines()
    all_exl = [i.strip() for i in all_exl] 

    #2. 全量字典和NER处理后的文件，clean the existing files，包括ner的边界，ner的顿号等问题，并保存
    logger.info("STAGE 1: File Pre-processing ")
    files = os.listdir(ner_training)

    for f in files:
        with open("%s%s"%(ner_training, f)) as file:
            print(f)
            dics = json.load(file)
            if f.islower():
                for x in dics:
                    ner = x["ner"]
                    x["ner"] = nerlencal(ner)
            else:
                temp_df = pd.DataFrame(dics)
                temp_df = temp_df[["paragraph","sentence","entity1","ner","seg"]]
                temp_df = temp_df.drop_duplicates(["paragraph","sentence","entity1"])
                dics = temp_df.to_dict("records")
                for x in dics:
                    ner = x["ner"]
                    x["ner"] = nerlencal(ner)
        with open("%s/%s"%(ner_training_cleaned, f),"w") as newf:
            json.dump(dics,newf)

    # 生成陌生新词词集
    logger.info("STAGE 2: Unseen dictionary generation ")
    with open(refined_dict) as f:
        refined_dictionaries = json.load(f)
    unseen_dictionaries = copy.deepcopy(refined_dictionaries)
    jieba_mapping = jieba_inverted
    type_list = [k for k in type_mapping.keys()]
    if dic_generation == "ON" or os.path.exists(unseen) == False:
        seen_words = []
        # display ts:
        for f in files:
            seen_split = []
            with open("%s%s"%(ner_training, f)) as file:
                print(f)
                dics = json.load(file)
                for seg in dics:
                    seg = seg["seg"]
                    for wtb in seg:
                        word, tag, bound = wtb
                        if word not in seen_split and tag in type_list:
                            seen_split.append(word)
            seen_words.append(seen_split)
            # print(len(seen_split))
        combined_seen = []
        for s in seen_words:
            for w in s:
                if w not in combined_seen:
                    combined_seen.append(w)
        unseen_words = {}
        for k,v in unseen_dictionaries.items():
            if k not in combined_seen:
                unseen_words[k] = v
        with open(unseen,"w") as fp:
            json.dump(unseen_words,fp)

    """实体合并"""
    logger.info("STAGE 3: Entity-Merging preprocessing")
    indications_dic_copy = []
    for f in files:
        with open("%s/%s"%(ner_training_cleaned,f)) as file:
            dics = json.load(file)
            dic_source = []
            for d in dics:
                source = f.replace("'","").split("_")[0]
                d["source"] = source
                dic_source.append(d)
            indications_dic_copy += dic_source
    ##有些字典里没有的实体类型，用NER替换
    indications_dic = copy.deepcopy(indications_dic_copy)
    notin_types = ["FW","DEG","AT","DUR","PSB","PT"]
    new_entities = []

    dicindex = 0
    for ix,dic in enumerate(indications_dic):
        seg_info = dic["seg"]
    #     candidates_seg=[i for i in seg_info if i[1] in seg_types]
    #     candidates_ner=[i for i in dic["ner"] if i[1] in ner_types]
        seg_bounds = [x[2] for x in seg_info]
        candidates_ner = [x for x in dic["ner"] if x[1] in notin_types]
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
            for i,si in enumerate(seg_info):
                if ner_pos == si[2]:
                    indications_dic[ix]["seg"][i][1]=ner[1]

    ner_types = list(re_dic.keys())
    # ["OBJ","EQM","DRU","PRE","TES","SUR","SYM","DIS","BAC","BFL"]
    seg_types = [jieba_mapping[i] for i in ner_types]
    desired_types = seg_types + ner_types

    # getting the suffix for diseases and symptoms
        # loading the suffix dictionary
    with open(suffix_dic) as f:
        suffix_dic = json.load(f)
    dissuf = []

    for k,v in suffix_dic.items():
        if v in ner_types:
            dissuf.append(k)     
    ## specify desired entity types and load suffix dictionaries
    ## merge the results for NER and dictionaries

    new_entities = []
    merged_results = []
    dicindex = 0
    for ix,dic in enumerate(indications_dic):
        seg_info = dic["seg"]
        candidates_seg = [i for i in seg_info if i[1] in seg_types]
        candidates_ner = [i for i in dic["ner"] if i[1] in ner_types]
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
            elif len(ner[0])>2:
                seg0 = [it[2][0] for it in seg_info]
                seg1 = [it[2][1] for it in seg_info]
                if ner_start in seg0 and ner_end in seg1:
                    merge_0 = seg0.index(ner_start)
                    merge_1 = seg1.index(ner_end)
                    if merge_0 == merge_1: ##same entity boundary but entity type is different, since we trust dictionary more, 
                        # these results are discarded
                        continue
                    new_entities.append(ner)
                    for to_del in range(merge_1+1,merge_0,-1): ## change
    #                     print(ix,seg_info[to_del-1][:2])
                        seg_info.pop(to_del-1)
    #                 ner[2][1]=ner[2][1]+1
                    seg_info.insert(merge_0,ner)
                else:
                    pass
        dicindex += 1
        dic["ind"] = dicindex
        for x,m in enumerate(ner_types):
            for si,s in enumerate(seg_info):
                if s[1] == m: 
                    seg_info[si][1] = seg_types[x]
                    print(seg_info[si])
        dic["seg"] = seg_info
        merged_results.append(dic)

    logger.info("STAGE 4: Entity-Merging")
    if merging_mode == "ON":
        pre_copy = words_combiner(merged_results,[["sr","sr"],["og","sr"],["ds","sr"]])
        merged_results = words_combiner(pre_copy,[["sr","ts"],["og","ts"],["ds","ts"],["ts","ts"]])
        eqsuffix = []
        tssuffix = []
        prsuffix = []
        srsuffix = []
        sysuffix = []
        dssuffix = []
        for k,v in suffix_dic.items():
            if v == "EQM":
                eqsuffix.append(k)
            elif v == "TES":
                tssuffix.append(k) 
            elif v == "SUR":
                srsuffix.append(k) 
            elif v == "SYM":
                sysuffix.append(k) 
            elif v == "DIS":
                dssuffix.append(k) 
                
        pre_copy = suffix_combiner(merged_results,tssuffix,desiredlist = ["om", "ts"],suffixtype="ts")
        merged_results = suffix_combiner(pre_copy,eqsuffix,desiredlist=["ts","bl","eq"],suffixtype="eq")
  
        pre_copy = suffix_combiner(merged_results,srsuffix,desiredlist=["ts","og","ds","sm","bl"],suffixtype="sr")
        po_series = [["og","ds"],["og","sm"],["ds","ds"],["sm","n","ds"],["og","n","ds"],\
                ["a","ds"],["og","n","sm"],["ds","a","ds"],["ds","k","ds"],\
                ["og","a","ds"],["og","a","sm"],["og","f","ds"],["og","uj","ds"],["og","f","sm"],["og","uj","sm"],\
                ["oj","oj"],["oj","dr"],["du","du"],["eq","eq"],["og","eq"],["ds","eq"],["og","eq"],\
                ["og","ts"]]
        merged_results = words_combiner(pre_copy, po_series)
        pre_copy = suffix_combiner(merged_results,sysuffix,desiredlist=["ds","sm","om"],suffixtype="sm")
        merged_results = suffix_combiner(pre_copy,dssuffix,desiredlist=["ds","sm","om"],suffixtype="ds")
        po_series = [["oj","sm"],["oj","ts"],["oj","sr"],["oj","oj"],["oj","sm"],["oj","ds"]]
        pre_copy = words_combiner(merged_results, po_series)
        po_series = [["oj","ds"],["sm","ds"],["dg","ds"],["dg","sm"],["ts","eq"],["og","eq"],["bc","eq"]]
        merged_results = words_combiner(pre_copy, po_series)


    # with open("processed_data/processing/training_NER_01.json","w") as f:
    #     json.dump(pre_copy2,f)

    """去掉某些不合适的实体"""
    logger.info("STAGE 5: problematic entity removal")
    # indications_copy_2 = copy.deepcopy(pre_copy)
    newly_merged = defaultdict(list)
    nontest = ["快速","化学","基因","快速","病理","其他","未知","游离","运动"]
    exl_symbols = ["、","；"]
    print("starting")
    cd = []
    md = []
    dd = []
    kbd = []

    for ici,j in enumerate(merged_results):
        if ici%1000000 == 0:
            print(ici)
        j_dict = {}
        for z1,z2 in zip(list(range(0,len(j["seg"]))),j["seg"]):
            j_dict[z1] = z2
        try:
            segs = [s[1] for s in j["seg"]]
            words = [s[0] for s in j["seg"]]
        except:
            print(j["seg"])
        j_copy = copy.deepcopy(j)
        ##因为后面删掉list元素后，或者插入元素后，index就不一致了，所以record number of poppped elements so far
        number_popped = 0
        display = False
        for i,wd in enumerate(words):
            if any(wd.endswith(nt) for nt in nontest):
    #             print("got one")
                j_copy["seg"][i][1] = "rm"
            elif [es for es in exl_symbols if es in wd]!=[] and len(wd)>=2:
                splitter = [es for es in exl_symbols if es in wd][0]
                splitted = wd.split(splitter)
                splitted_test = [len(i) for i in splitted if len(i)>0]
                if np.min(splitted_test) <= 2:
                    continue
                display = True
                splitted_tag = j["seg"][i][1]
                starter = j["seg"][i][2][0]
                ender = j["seg"][i][2][1]
    #             splitted_pos=
                j_copy["seg"].pop(i+number_popped)
                number_popped -= 1
    #             ind_to_pop=i+number_popped
                for s_i,sp in enumerate(splitted):
                    if sp == "":
                        sb_pos=[starter,starter+1]
                        number_popped += 1
                        j_copy["seg"].insert(i+number_popped,[splitter,"rm",sb_pos]) 
                        starter += 1
    #                     ind_to_pop=i+number_popped
                        continue
                    spl = len(sp)
                    pl_pos = [starter,starter+spl]
                    starter += spl
                    number_popped += 1
                    j_copy["seg"].insert(i+number_popped,[sp,splitted_tag,pl_pos]) 
                    if s_i != len(splitted) - 1:
                        if len(splitted[s_i+1]) > 0:
                            number_popped += 1
                            sb_pos = [starter,starter+1]
                            j_copy["seg"].insert(i+number_popped,[splitter,"rm",sb_pos]) 
                            starter += 1
        
    #                 ind_to_pop=i+number_popped
        if display == True:
            print("...so far:",ici)
        j["seg"] = j_copy["seg"]
        if j["source"] == "c":
            cd.append(j)
        elif j["source"] == "d":
            dd.append(j)       
        elif j["source"] == "m":
            md.append(j)
        else:
            kbd.append(j)  

    for xf in (cd, dd, md, kbd):
        indications_df = pd.DataFrame(xf)
        indications_df_2 = indications_df.drop_duplicates(["source","paragraph","sentence","entity1"])
        indications_dic = indications_df_2.to_dict("records")
        print(len(indications_dic))
        filename = xf[0]["source"]
        with open("{}/{}_all.json".format(ner_training_reformatted, filename),"w") as f:
            json.dump(indications_dic,f)

    """json转化为训练数据，并且训练测试数据划分分别存储"""
    logger.info("STAGE 5: train/test data reformatting and saving")
    # indications_dic[300000:300100]
    filename = "{}/{}_{}.txt"
    test_seeds = []
    # condname="../activeNER/ner-medical-dev1/data/new_training/{}_cond_v2.txt"
    for f in os.listdir(ner_training_reformatted):
        file_head = f.split("_")[0]
        print(file_head)
        with open("{}/{}".format(ner_training_reformatted, f), "r") as sf:
            indications_dic = json.load(sf)
        with open(filename.format(ner_training_final, file_head,"training"),"w") as tf,open(filename.format(ner_training_final, file_head,"validation"),"w") as vf, \
        open(filename.format(ner_training_final, file_head,"test"),"w") as ef:
            for i,xele in enumerate(indications_dic):
                if i%40000 == 0:
                    print(i)
                if i%80 == 0:
                    get_BI_word(xele,vf)
                elif i%90 == 0:
                    get_BI_word(xele,ef)
                    test_seeds.append(xele)    
                else:
                    get_BI_word(xele,tf)

#     with open("{}/test_seed.json".format(ner_training_final),"w") as f:
#         json.dump(test_seeds,f)
    with open('{}/test_seed.txt'.format(ner_training_final),"w") as ef:
        for i,xele in enumerate(test_seeds):
            get_BI_word(xele,ef)
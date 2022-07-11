"""
主要功能：1. 利用字典（存储在dictionaries路径中）对结构化的文本数据进行预标注，以便后续生成NER训练数据。同时可以通过其中的参数选择是否同时利用训练好的NER模型来辅助标注。
          分别对 临床指南，操作手册，药品说明书， 资料库进行处理，得到json文件。其中ner结果和字典分词结果分别存储为dic的不同的key中。
          每条句子生成的形式为 (注意这里seg和ner的结果在边界的index上可能有点不一致，一个index包含最后字符一个不包括，后续还要处理)：
                    'book': '心血管分册.txt',
                    'location': '|第一章心力衰竭|第一节慢性心力衰竭',
                    'paragraph': 'clinical',
                    'sentence': '【临床表现】常见症状①呼吸困难:肺淤血所致,依病情不同可出现劳力性呼吸困难,夜间阵发性呼吸困难,甚至端坐呼吸',
                    'entity1': '慢性心力衰竭',
                    'seg': [['【', 'x', [0, 1]],
                      ['临床表现', 'i', [1, 5]], ..., 
                      ['呼吸困难', 'ds', [11, 15]],
                      ...]],
                    'ner': [['呼吸困难', 'SYM', [11, 14], [0.7744848407083914, 0.9460845777994957]],
                      ['肺淤血', 'DIS', [16, 18], [0.7440568890117892, 0.9224377721677368]],
                      ['劳力性呼吸困难', 'SYM', [30, 36], [0.920548668509957, 0.9530912794173739]], ...]]
          2. 药品说明书的人群禁忌知识直接抽取并且存储在results文件夹中。
使用场景：模型训练。用于训练NER模型前的初步训练数据生成
"""

import pandas as pd
import numpy as np
import os
import re
import json
import jieba
import jieba.posseg
from collections import defaultdict
from utils import prefix_tool, Utillist, Model_selector
import yaml
from loguru import logger
import time

with open('config.yaml', 'rb') as fp:
    dic_file = yaml.safe_load(fp)
    ## when ner model is enabled during pre-labeling task, 
    # sequences will be labeled by both dictionaries and ner model.
    # otherwise we label the sequences solely according to dictionaries, this is way to go when we don't have
    # the relevant ner model yet (when we have new ontology schema/entities etc)
    ner_on = dic_file['setting']['ner_on']
    clinicals = dic_file['corpus']['clinicals']
    manuals = dic_file['corpus']['manuals']
    instructions = dic_file['corpus']['instructions']
    knowledge_base = dic_file['corpus']['knowledge_base']
    RE = dic_file['ontologies']['RE']
    stop_list = dic_file['dictionaries']['stop']
    full_dict = dic_file['dictionaries']['full']
    refined_dict = dic_file['dictionaries']['refined']
    model_path = dic_file['model_path']
    preprocessed_clinical = dic_file['data_files']['preprocessed_clinical']
    preprocessed_manual = dic_file['data_files']['preprocessed_manual']
    ## 训练数据存储的文件夹
    ner_training = dic_file['training_data']['ner_training']
    full_jieba = dic_file['dictionaries']['full_jieba']
    group_results = dic_file['results']['group']
def multi_level_dict():
    return defaultdict(multi_level_dict)
        
if __name__ == "__main__":
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path =  'logs/preprocessing_'
    logfile = log_path + rq + '.log'
    logger.add(logfile, backtrace=True, diagnose=True, rotation='3 days', retention='2 months')
    logger.info("STAGE 0: Model Initialization ")
    if ner_on == True:
        Mselect = Model_selector()
        cn = Mselect.selection("{}/{}".format(model_path, "params_o_cnn_a.pkl"))
    jieba.load_userdict(full_jieba)
    """1. 知识库：其他"""
    print("=== processing knowledge base, step I ======")
    logger.info("== processing knowledge base, step I ====== ")
    filelist = ['surgeries', 'labs', 'examinations']
    for f in filelist:
        operation_df = pd.read_csv("{}/{}.csv".format(knowledge_base, f))
        for colname in operation_df.columns[1:]:
            indications = []
            totallen = len(operation_df)
            indications_ner = []
            indicator = 0
            for me,p in zip(operation_df.measure, operation_df[colname]):
                indicator += 1
                if type(p) is not str:
                    continue
                sub_lines = re.split('[。?？!！\t ]', p)
                for sent in sub_lines:
                    if len(sent) <= 3:
                        continue
                    seg_list = jieba.posseg.cut(sent)
                    word_cut = []
                    p0 = 0
                    p1 = 0
                    for s in seg_list:
                        p1 += len(s.word)
                        position = [p0,p1]
                        p0 = p1
                        word_cut.append([s.word,s.flag,position])
                    my_dict = multi_level_dict()
                    my_dict["paragraph"] = colname
                    my_dict["sentence"] = sent
                    my_dict["entity1"] = me
                    my_dict["seg"] = word_cut
                    if ner_on == True:
                        result_dict = cn.predict_oneline(input_str = sent)[0]
                        items=[]
                        for item in result_dict.values():
                            item.pop(2)
                            items.append(item)
                        my_dict["ner"] = items
                    else:
                        my_dict["ner"] = []
                    indications.append(my_dict)
                    if indicator%1000 == 0:
                        logger.info("{}:{}".format(colname, indicator/totallen))
            with open("data/Juli/%s_%s.json"%('ki', colname),"w") as fb:
                    json.dump(indications,fb)


    """2. 知识库：疾病"""
    logger.info("processing knowledge base step II")
    operation_df = pd.read_csv("{}/diseases.csv".format(knowledge_base))
    for colname in operation_df.columns[1:]:
        indications = []
        totallen = len(operation_df)
        indications_ner=[]
        indicator = 0
        for me,p in zip(operation_df.disease, operation_df[colname]):
            indicator += 1
            if type(p) is not str:
                continue
            sub_lines = re.split('[。?？!！\t ]', p)
            for sent in sub_lines:
                if len(sent) <= 3:
                    continue
                seg_list = jieba.posseg.cut(sent)
                word_cut = []
                p0 = 0
                p1 = 0
                for s in seg_list:
                    p1 += len(s.word)
                    position = [p0,p1]
                    p0 = p1
                    word_cut.append([s.word,s.flag,position])
                my_dict = multi_level_dict()
                my_dict["paragraph"] = colname
                my_dict["sentence"] = sent
                my_dict["entity1"] = me
                my_dict["seg"] = word_cut
                items = []
                if ner_on == True:
                    result_dict = cn.predict_oneline(input_str = sent)[0]
                    items = []
                    for item in result_dict.values():
                        item.pop(2)
                        items.append(item)
                    my_dict["ner"] = items
                else:
                    my_dict["ner"] = []
                indications.append(my_dict)
                if indicator%1000 == 0:
                    logger.info("{}:{}".format(colname, indicator/totallen))
        with open("%s%s_%s.json"%(ner_training, 'kd', colname),"w") as fb:
                json.dump(indications,fb)

    """3. 临床指南"""
    logger.info("processing clinicals")
    operation_df = pd.read_csv(preprocessed_clinical)
    for colname in operation_df.columns[4:]:
        indications = []
        totallen = len(operation_df)
        indications_ner = []
        indicator = 0
        for me,p,l,b in zip(operation_df.disease,operation_df[colname],operation_df.location,operation_df.book_name):
            indicator += 1
            if type(p) is not str:
                continue
            sub_lines = re.split('[。;?？!！；\t ]', p)
            for sent in sub_lines:
                if len(sent) <= 3:
                    continue
                seg_list = jieba.posseg.cut(sent)
                word_cut = []
                p0 = 0
                p1 = 0
                for s in seg_list:
                    p1 += len(s.word)
                    position = [p0,p1]
                    p0 = p1
                    word_cut.append([s.word,s.flag,position])
                my_dict = multi_level_dict()
                my_dict["book"] = b
                my_dict["location"] = l
                my_dict["paragraph"] = colname
                my_dict["sentence"] = sent
                my_dict["entity1"] = me
                my_dict["seg"] = word_cut
                if ner_on == True:
                    result_dict = cn.predict_oneline(input_str = sent)[0]
                    items = []
                    for item in result_dict.values():
                        item.pop(2)
                        items.append(item)
                    my_dict["ner"] = items
                else:
                    my_dict["ner"] = []
                indications.append(my_dict)
            if indicator%1000 == 0:
              logger.info("{}:{}".format(colname, indicator/totallen))
        with open("%s%s_%s.json"%(ner_training, 'c', colname),"w") as fb:
                json.dump(indications, fb)

    """4. 操作手册"""
    logger.info("processing manuals")
    operation_df = pd.read_csv(preprocessed_manual)
    logger.info(operation_df.columns[4:])
    for colname in operation_df.columns[4: ]:
        indications = []
        totallen = len(operation_df)
        indications_ner = []
        indicator = 0
        for me,p,l,b in zip(operation_df.measure,operation_df[colname],operation_df.location,operation_df.book_name):
            indicator += 1
            # if indicator >= 50:
            #     break
            if type(p) is not str:
                continue
            sub_lines = re.split('[。;?？!！；\t ]', p)
            for sent in sub_lines:
                if len(sent) <= 3:
                    continue
                seg_list = jieba.posseg.cut(sent)
                word_cut = []
                p0 = 0
                p1 = 0
                for s in seg_list:
                    p1 += len(s.word)
                    position = [p0,p1]
                    p0 = p1
                    word_cut.append([s.word,s.flag,position])
                my_dict = multi_level_dict()
                my_dict["book"] = b
                my_dict["location"] = l
                my_dict["paragraph"] = colname
                my_dict["sentence"] = sent
                my_dict["entity1"] = me
                my_dict["seg"] = word_cut
                if ner_on == True:
                    result_dict = cn.predict_oneline(input_str = sent)[0]
                    items = []
                    for item in result_dict.values():
                        item.pop(2)
                        items.append(item)
                    my_dict["ner"] = items
                else:
                    my_dict["ner"] = []
                indications.append(my_dict)
            if indicator%1000 == 0:
               logger.info("{}:{}".format(colname, indicator/totallen))
        with open("%s%s_%s.json"%(ner_training, 'm', colname),"w") as fb:
                json.dump(indications,fb)

    """5. 药品说明书"""
    logger.info("processing instructions")
    operation_df = pd.read_csv(instructions)
    colnames = ['COMPONENT', 'INDICATION', 'TABOO', 'BAD_REACTION']
    logger.info(colnames)
    for colname in colnames:
        logger.info("starting to prepare: {}".format(colname))
        if colname == 'ATTETION_ITEM':
            continue
        indications = []
        totallen = len(operation_df)
        indications_ner = []
        indicator = 0
        for me,p in zip(operation_df.PRODUCT_NAME_CN,operation_df[colname]):
            indicator += 1
            # if indicator >= 500:
            #     break
            if type(p) is not str:
                continue
            sub_lines = re.split('[。?？!！\t ]', p)
            for sent in sub_lines:
                if len(sent) <= 3:
                    continue
                seg_list = jieba.posseg.cut(sent)
                word_cut = []
                p0 = 0
                p1 = 0
                for s in seg_list:
                    p1 += len(s.word)
                    position = [p0,p1]
                    p0 = p1
                    word_cut.append([s.word,s.flag,position])
                my_dict = multi_level_dict()
                my_dict["paragraph"] = colname
                my_dict["sentence"] = sent
                my_dict["entity1"] = me
                my_dict["seg"] = word_cut
                if ner_on == True:
                    result_dict = cn.predict_oneline(input_str = sent)[0]
                    items = []
                    for item in result_dict.values():
                        item.pop(2)
                        items.append(item)
                    my_dict["ner"] = items
                else:
                    my_dict["ner"] = []
                indications.append(my_dict)
            if indicator%4000 == 0:
                logger.info("{}:{}".format(colname, indicator/totallen))
        with open("%s%s_%s.json"%(ner_training, 'd', colname),"w") as fb:
                json.dump(indications,fb)

    """6. 药品说明书禁忌部分，直接获取三元组，不需要后续模型处理"""

    def get_group_caution(column,sentence,gname):
        ## 获取禁忌关系三元组
        grptri = []
        for i,j,k in zip(groups.PRODUCT_NAME_CN,groups[column],groups[sentence]):
            if str(j) != "nan":
                grptri.append([i,gname,j,k])
        return(grptri)

    groups = pd.read_csv(instructions)
    colnames = ['WOMAN_MEDICINE', 'CHILDREN_MEDICINE', 'AGEDNESS_MEDICINE']
    prohibited = ["禁用","不适用","禁止","忌"]
    notrecommended = ["不宜","慎"]
    caution = ["致","易发","指导","减量","调整","较小","减少","酌","注意","低剂"]
    elderly = []
    for colname in colnames:
        group_contrain = []
        for j in groups[colname]:
            j = str(j)
            taboo = ''
            for t in caution:
                if t in str(j): 
                    taboo = "调整用量"
                    break
            for t in notrecommended:
                if t in str(j): 
                    taboo = "慎用人群"
                    break
            for t in prohibited:
                if t in str(j): 
                    taboo = "禁用人群"
                    break
            group_contrain.append(taboo)
        if "WOMAN" in colname:
            newcol = "pregnancy"
        elif "CHILDREN" in colname:
            newcol = "children"
        elif "AGEDNESS" in colname:
            newcol = "elderly"
        groups[newcol] = group_contrain
        
                
    pregs = get_group_caution("pregnancy","WOMAN_MEDICINE","孕妇")
    kids = get_group_caution("children","CHILDREN_MEDICINE","儿童")
    elds = get_group_caution("elderly","AGEDNESS_MEDICINE","老人")
        
    taboo_verb = ["禁用","慎用","谨慎","不适用","禁止","忌","不宜"]
    groups = ["孕妇","哺乳","儿童","老人","妊娠","新生儿","婴儿","幼儿","小儿"]
    trigger = "过敏"
    verbs = ["引起","诱发","导致","造成","引发"]
    contra_all = []
    triples = []
    group_df = pd.DataFrame(pregs+kids+elds)
    group_df = group_df.drop_duplicates()
    group_df.columns = ["entity_1","entity_2","relation","sentence"]
    group_df["entity_type_1"] = "DRU"
    group_df["entity_type_2"] = "GRP"
    group_df["domain"] = "KBMS"
    group_df["entity_type_1"] = "DRU"
    group_df["entity_type_2"] = "GRP"
    group_df["domain"] = "KBMS"
    group_df.loc[group_df.relation=="调整用量","relation"] = "慎用人群"
    group_df.to_csv(group_results)     
    # #                 print(name,"人群"+rel,g)
    # #         print(rel,contra,c)
    #     except:
    # #         print("sssss")
    #         pass
    # #         print("not found{}".format(c))

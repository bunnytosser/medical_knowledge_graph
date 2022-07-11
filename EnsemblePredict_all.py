"""
主要功能：1. 利用测试集，根据评价指标生成各个模型评价矩阵
          2. 再测试集上对各个模型的预测进行集成，并且也进行相关的评估
          多语料模型分别再多个测试集评估
          多集成方法的评估，包括：
          1. 取各个模型预测的并集边界
          2. 对各个模型根据其在各类实体的表现评估生成权重矩阵，对预测结果进行加权平均预测
          3. 对个模型预测结果，首先取边界并集合并成大实体，然后取至少有两个模型支持的实体（只要模型预测的子字符串在大实体边界内就算一个）
使用场景：模型训练。每当模型更新后即可对各个模型进行评估生成评估的权重矩阵。

6. ensemblePrediction_all.py  # 结合评价矩阵，利用多语料多个模型对全量语料进行标注并进行相关的集成。
主要功能：1. 多语料训练出来的模型分别对全量数据进行标注预测。
         2. 利用模型评估矩阵，对标注的结果利用多种方式进行集成。
         3. 对置信度等进行计算整理并存储。
        
        集成方式包括：1）取各个模型预测的并集边界
        2） 对各个模型根据其在各类实体的表现评估生成权重矩阵，对预测结果进行加权平均预测
        3） 对个模型预测结果，首先取边界并集合并成大实体，然后取至少有两个模型支持的实体（只要模型预测的子字符串在大实体边界内就算一个）
        存储的格式：
            {'book': '器官移植分册.txt',
            'location': '|第1章肾移植|第十六节移植肾切除术',
            'paragraph': 'indications',
            'sentence': '2.移植肾破裂,无法修补者器官移植分册3.血管并发症治疗失败者',
            'entity1': '移植肾切除术',
            'seg': [['2.', 'm', [0, 2]],
              ['移植肾破裂', 'ds', [2, 7]],
              [',', 'x', [7, 8]],
              ['无法', 'n', [8, 10]],
              ['修补者', 'nr', [10, 13]],
              ['器官移植', 'sr', [13, 17]],
              ['分册', 'v', [17, 19]],
              ['3.', 'm', [19, 21]],
              ['血管', 'og', [21, 23]],
              ['并发症', 'ds', [23, 26]],
              ['治疗', 'v', [26, 28]],
              ['失败者', 'n', [28, 31]]],
            'ner': [['器官移植', 'SUR', [13, 17], [0.837152347912683, 0.9458935400001512]],
              ['血管', 'ORG', [21, 23], [0.7998752802310558, 0.8750970425641398]],
              ['并发症', 'DIS', [23, 26], [0.852185656717616, 0.9363446760837277]]],
            'source': 'm',
            'ind': 90046,
            'ensemble': [['移植', 'SUR', [2, 4]],
              ['器官移植', 'SUR', [13, 17]],
              ['血管', 'ORG', [21, 23]],
              ['并发症', 'DIS', [23, 26]],
              ['肾破裂', 'DIS', [4, 7]]],
            'entity1_type': 'SUR'}
          集成结果通过ensemble的key保存，而ner的key则保存全量语料模型"a"的预测结果。
"""
import pandas as pd
import numpy as np
import os
import re
import json
import jieba
import jieba.posseg
from multi_source_ner import ChineseNER
import codecs
from ner.ner_evaluation.ner_eval import collect_named_entities
from ner.ner_evaluation.ner_eval import compute_metrics
from ner.ner_evaluation.ner_eval import compute_precision_recall_wrapper
from ner.ner_evaluation.ner_eval import Evaluator
import json
from collections import defaultdict
import argparse
import yaml
import time
from loguru import logger
from multiprocessing import Pool
import math

# sentence = "5.瓣膜置换术后心内膜炎,感染严重,药物不易控制,引起人工瓣功能障碍或瓣周漏、瓣周脓肿等。需利用人工导管进行卵巢切除术，并服用阿莫西林，清咽滴丸，三位地黄片等药物并施加电热治疗。对于肠道有严重沙门氏菌感染的患者，需住院观察并输液。"
# sentence2 = "(1)缺血性或非缺血性心肌病(2)充分抗心力衰竭药物治疗后,NYHA心功能分级仍在Ⅲ级或不必卧床的Ⅳ级(3)窦性心律4)左心室射血分数≤35%30第2章心脏起搏(5)左心室舒张末期内径≥55mm(6)QRS波时限≥120ms伴有心脏运动不同步2.Ⅱ类适应证(1)Ⅱa类适应证①充分药物治疗后NYHA心功能分级好转至Ⅱ级,并符合Ⅰ类适应证的其他条件②慢性心房颤动患者,合乎I类适应证的其他条件,可结合房室结射频消融行CRT治疗,以保证夺获双心室(2)Ⅱb类适应证①符合常规心脏起搏适应证并心室起搏依赖的患者,合并器质性心脏病NYHA心功能Ⅲ级及以上"
# model_indicator = False
# print(sentence)
def callback_success(value):
    global success_num
    success_num += 1
    logger.info('success num: {}'.format(success_num))
def callback_error(e):
    raise e
#    _, _, tb = sys.exc_info()
#    print(traceback.format_list(traceback.extract_tb(tb)[-1:])[-1])

def get_model(modname):
    """简化模型名称 在字典存储"""
    modname = modname.split(".")[0]
    if "aug" in modname:
        clean = "_".join(modname.split("_")[-2:])
    elif "v2" not in modname:
        clean = modname.split("_")[-1]
    elif "v2" in modname:
        clean = "".join(modname.split("_")[-2:])
        clean = "".join([clean[0],clean[-1]])
    return(clean)


def post_processing(final_result):
    new_result = []
    lagged =""
    for i,w in enumerate(final_result):
        if w == "O":
            new_result.append("O")        
        elif w == lagged:
            new_result.append("I-%s"%w)
        else:
            new_result.append("B-%s"%w)    
        lagged = w
    return(new_result) 

def result_cleaning(dic):
    removal_list = []
    for model, slist in dic.items():
        for i, s in enumerate(slist):
            if s == []:
                
                removal_list.append(i)
    if len(removal_list) == 0:
        return(dic)
    else:

        removal_list = list(set(removal_list))
        removal_list.sort(reverse=True)
       # print(removal_list)
        for model, slist in dic.items():
            for rl in removal_list:
                slist.pop(rl)
            dic[model] = slist
        return(dic)





def predict_processing(model_name, sentence):
    test1, probs_path = model_name.predict_oneline(input_str = sentence)
    results = list(test1.values())
    sentencelist = list(sentence)
    taglist = ["O" for x in sentencelist]
    for r in results:
        start_idf = 0
        for rr in np.arange(r[3][0], r[3][1] + 1):
            if start_idf == 0:
                taglist[rr] = 'B-' + r[1]
            else:
                taglist[rr] = 'I-' + r[1]
            start_idf += 1
    return(taglist, probs_path)


        
def most_frequent(List):
    return max(set(List), key = List.count)

def get_word_ind(list_se, idx):
    found_id = -1
    for i,sublist in enumerate(list_se):
        if idx >= sublist[0] and idx < sublist[1]:
            found_id = i
    return(found_id)

def get_positions(es):
    ## return the positions of the start/end index for entities and the corresponding type, probabilities
    positions = []
    len_seq = len(es) - 1
    init_pos = [0, 0]
    waiting = False
    c_types = []
    prob_seqs = []
    prob_seq = []            
    for ie, element_tuple in  enumerate(es):
        (element, prob) = element_tuple
        if element.split("-")[0] == "I":
            if ie != len_seq: 
                prob_seq.append(prob)
                waiting = True
            else: # end of the sentence and still inside an entity
                init_pos[1] = ie + 1
                c_types.append(type_waiting)
                prob_seq.append(prob)
                positions.append(init_pos)  

                prob_seqs.append([np.max(prob_seq), np.min(prob_seq)])          
        elif ie == len_seq and element == 'O': #end of the sentence and not inside an entity
            if waiting == True: # if the previous char is the end of the entity
                init_pos[1] = ie
                c_types.append(type_waiting)
                prob_seqs.append([np.max(prob_seq), np.min(prob_seq)]) 
                positions.append(init_pos)
            else:
                continue
        elif element.split("-")[0] == "B": # at the beginning of an entity

            if waiting == True: # at the beginning of a new entity right after an old one 
                init_pos[1] = ie
                c_types.append(type_waiting)
                positions.append(init_pos)
                # recorder[str(init_pos[0]) + "|" + str(init_pos[1])] = []
                init_pos = [0, 0]
                prob_seqs.append([np.max(prob_seq), np.min(prob_seq)]) 
                prob_seq = [prob]        
                waiting = True
            else:
                prob_seq = [prob]
            type_waiting = element.split("-")[1]
            init_pos[0] = ie

        elif element == 'O':
            if waiting == True or ie == len_seq:
                if ie == len_seq:
                    ie = ie + 1
                waiting = False
                init_pos[1] = ie
                c_types.append(type_waiting)
                positions.append(init_pos)
                # recorder[str(init_pos[0]) + "|" + str(init_pos[1])] = []
                init_pos = [0, 0]
                prob_seqs.append([np.max(prob_seq), np.min(prob_seq)]) 
                prob_seq = [prob]     
            else:
                continue
    return(c_types, positions, prob_seqs)  

# test_datasets = []
# print(test_datasets)
def multi_labeling(idx, hs, dslength, fname, ent_dic):
    global c_collections, data
    # assigning entity types

    cs = list(hs["sentence"])
    s = hs["sentence"]
    # cs, t, s = hs   
    pred_dict = {}

    ## get the model predctions and convert them into the [(tag, prob),...] sequence format
    for pa, mod in multi_models.items():
        # try:
        c_results, probs_path = predict_processing(mod, s)

        # except:
        #     c_results = []
        
        label_prob = [(i1, i2) for i1,i2 in zip(c_results, probs_path)]
        pred_dict[pa] = label_prob
        #print(pred_dict)
        #pred_dict["probs_path"] = probs_path
    label_ensembles = {}
    # logger.info("1. 用多种模型预测：complete")
    # logger.info("2. 集成处理: starting")
    """交集集成处理"""

    # if args.ensemble == "pre":
    prob_intersect = [i for (j,i) in pred_dict["a"]]
    for no in range(len(pred_dict["a"])):
        label_ensembles[no] = defaultdict(int)

    for model, lseq in pred_dict.items():
        if model == "original": continue
        for n, (label, lprob) in enumerate(lseq):
            if label == 'O': continue
            else:
                etype = label.split("-")[1]
                if etype not in types:
                    continue
                label_ensembles[n][etype] += weights[model][etype]
                prob_intersect[n] = np.min([prob_intersect[n], lprob])
    

    final_result = []

    for k, res in label_ensembles.items():
        if res == {}:
            final_result.append("O")
            prob_intersect.append(0)

        else:
            res["O"] = 1 - float(np.sum(list(res.values())))
            max_key = max(res, key=res.get)
            if res[max_key] >= 0.2:
                final_result.append(max_key)
                prob_intersect.append(res[max_key])
            else:
                final_result.append("O")
                prob_intersect.append(0) 


    ensemble_votes = post_processing(final_result)
    ensemble_intersect = [(i,j) for (i,j) in zip(ensemble_votes, prob_intersect)]

    """并集集成法"""
    ensemble_union = []
    prob_union = []

    for (xi, xi_p) in pred_dict["a"]:
        prob_union.append(xi_p)
        if xi == 'O': 
            ensemble_union.append(xi)
        else:
            etype = xi.split("-")[1]
            ensemble_union.append(etype)


    for model, lseq in pred_dict.items():
        if model == "original": continue
        for n, (label, lprob) in enumerate(lseq):
            if label == 'O': continue
            else:
                label = label.split("-")[1]
                if label != ensemble_union[n] and ensemble_union[n] == 'O':
                    ## 并集的概率取并的较大者
                    prob_union[n] = np.max([prob_union[n], lprob])
                    ensemble_union[n] = label

    ensemble_union = post_processing(ensemble_union) 
    ensemble_union = [(i,j) for (i,j) in zip(ensemble_union, prob_union)]
    pred_dict["ensemble_union"] = ensemble_union            
    pred_dict["ensemble_votes"] = ensemble_intersect

    # logger.info("2. 集成处理: finished")
    # logger.info("3. 投票: starting")
    """并集后交集投票集成法：先并集 后根据数量筛选(并集的ent type recall 
    precision was not reduced comparing comparing to "all" model)"""

    ensemble_majority = {}
    for no in range(len(pred_dict["ensemble_union"])):
        ensemble_majority[no] = defaultdict(list)
    
    # 构建一个记录并集标签后 词语起始位置的列表行字典，方便后续往每个合并后词 append 模型结果
    # recorder = {}
    positions = []
    len_seq = len(pred_dict["ensemble_union"]) - 1
    init_pos = [0, 0]
    waiting = False
    c_types = [] 
    
    c_types, positions, prob_seqs = get_positions(pred_dict["ensemble_union"])
#           print("prob seqs", prob_seqs)

    frequency_counts = defaultdict(list)
    prob_combined = [i for (j,i) in pred_dict["a"]]
    for model, lseq in pred_dict.items():
        if model == "original" or 'ensemble' in model: continue
        for n, (label, lprob) in enumerate(lseq):
            if label == 'O': continue
            else:
                etype = label.split("-")[1]
                combined_index = get_word_ind(positions, n) # get the index of the combined words the current character is located in
                prob_combined[n] = np.max([prob_combined[n], lprob])
                if combined_index >= 0:
#                        print(positions, combined_index)
                    frequency_counts[combined_index].append(model)
                else:
                    continue
            
    ## 形式复原
    voted_combined = []
    models_combined = []
    types_combined = []
    boundaries_combined = []

    # print(frequency_counts)
    for cbidx,mdls in frequency_counts.items():
        if len(set(mdls)) >= 2:
            voted_combined.append([positions[cbidx], c_types[cbidx]])
            models_combined.append(list(set(mdls)))
            types_combined.append(c_types[cbidx])
    
    voted_seq = ["O"] * (len_seq + 1)
    for vc in voted_combined:
        start = vc[0][0]
        end = vc[0][1]
        boundaries_combined.append([start, end])
        tp = vc[1]
        voted_seq[start] = "B-" + tp
        for every_char in range(start + 1, end):
            voted_seq[every_char] = "I-" + tp
    
    voted_seq = [(i,j) for i,j in zip(voted_seq, prob_combined)]
#      print("voted strong", voted_seq)
    # a_types, a_positions, prob_seqs = get_positions(pred_dict["a"])
    # pred_dict["a_types"] = a_types
    # pred_dict["a_positions"] = a_positions
    pred_dict["ensemble_strong"] = voted_seq
    pred_dict["ensemble_boundary"] = boundaries_combined
    pred_dict["sentence"] = cs
    pred_dict["ensemble_models"] = models_combined
    pred_dict["ensemble_types"] = types_combined
    # print(data[idx], data[idx].keys())
    pred_dict["ind"] = data[idx]["ind"]
    c_collections.append(pred_dict)
    ensemble_sequence = []
    for eb, et in zip(boundaries_combined, types_combined):
        es = "".join(cs[eb[0]: eb[1]])
        ensemble_sequence.append([es, et, eb])
    data[idx]["ensemble"] = ensemble_sequence

    entity1_type = ""
    if 'kd_all' in fname:
        if data[idx]["source"] == 'examinations':    
            data[idx]["entity1_type"] = "TES"
        elif data[idx]["source"] == 'labs':    
            data[idx]["entity1_type"] = "TES"  
        elif data[idx]["source"] == 'surgeries':    
            data[idx]["entity1_type"] = "SUR"  
        elif data[idx]["source"] == 'kd':    
            data[idx]["entity1_type"] = "DIS"  
    elif "c_all" in fname:
        data[idx]["entity1_type"] = 'DIS'
    elif "d_all" in fname:
        data[idx]['entity1_type'] = 'DRU'
    elif "m_all" in fname:
        data[idx]['entity1_type'] = 'SUR'
        e1type = ent_dic.get(data[idx]["entity1"])
        data[idx]["entity1_type"] = e1type
    if idx%1000 == 0:
        logger.info("***用多种模型预测：# {}/{} finished, around {} %".format(idx,dslength, round(100*idx/dslength),2))  

def main():
    global evaluation_path, ner_results, multi_models, c_collections, data, types, weights, full_dict
    pool_size = 10
    with open('config.yaml', 'rb') as fp:
        dic_file = yaml.safe_load(fp)
        essential_dic = dic_file['ontologies']['RE']
        types = list(essential_dic.keys())
        nonessential_dic = dic_file['ontologies']['RE_nonessential']
        tag_mapping = {**essential_dic, **nonessential_dic}
        model_path = dic_file['model_path']
        desirable_models = dic_file['prediction_model']
        ner_training_reformatted = dic_file['training_data']['ner_training_reformatted']
        ner_training_final = dic_file['training_data']['ner_training_final']
        evaluation_path = dic_file['evaluation_path']
        ner_results = dic_file['results']['ner_results']
        version_no = dic_file['model_version']
        full_dict = dic_file['dictionaries']['full']
# initializting all models
    multi_models = {}
    for model_name in os.listdir(model_path):
        if model_name in desirable_models:
#             print("!!!!!!!!!!!!loading model: {}/{}".format(model_path, model_name))
            ner_model = ChineseNER("predictonly", "{}/{}".format(model_path, model_name))
            multi_models[get_model(model_name)] = ner_model

    test_datasets = [x for x in os.listdir(ner_training_final) if "a_test" in x]
    with open(full_dict) as juned:
        ent_dic = json.load(juned)
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path =  'logs/labeling_'
    logfile = log_path + rq + '.log'
    logger.add(logfile, backtrace=True, diagnose=True, rotation='3 days', retention='2 months')


    num_models = len(multi_models)
    with open("{}/model_weights.json".format(ner_results), "r") as f:
        weights = json.load(f)
    logger.info("model weights loaded")
    results_full = []
    simplified_models = [get_model(m) for m in desirable_models]
    logger.info(simplified_models)
    evaluation_metrics = {}
    combined_results = []
    with open("{}/eval_dic.json".format(evaluation_path), "r") as evald:
        evaluation_metrics = json.load(evald)

    ## merging seperate json files into one
    for fname in os.listdir(ner_training_reformatted):
        fname_abbrev = fname.split(".")[0]
        if fname.endswith("json") == False: continue
        with open('{}/{}'.format(ner_training_reformatted,fname)) as fdata:
            data = json.load(fdata)
       # data = data[:100]
        dslength = len(data)
        logger.info("开始多模型预测...\n当前进行预测的数据集是：{}, 待预测数据量为:{}条，使用模型数量为:{}".format(fname, dslength,len(multi_models)))
        c_collections = []

        segs = []
        for idx, hs in enumerate(data):

            ##对每一条数据都进行多模型预测和集成
            
            multi_labeling(idx, hs, dslength, fname, ent_dic)


        logger.info("**用多种模型预测：finished") 

        converted = defaultdict(list)

        ### c_collections中的每一个元素 为:{'模型':[(char1, prob1),(char2,prob2),...],'模型2':[...],...}形式的字典。
        for v in c_collections:
            for m,l in v.items():
                converted[m].append(l)

        converted = result_cleaning(converted)
        # with open("{}/converted_{}_{}.json".format(ner_results, fname.split(".")[0], version_no), "w") as nrw:
        #     json.dump(converted, nrw)

        print("keys of converted", converted.keys())
        ## converted: 每一个模型作为key，该模型预测的全部序列的类型和置信度
        
        logger.info("**后处理: starting")  
        for no_sample in range(len(converted["a"])):
            combined_result = {}
            sentence_original = converted["sentence"][no_sample]
            full_sent = "".join(sentence_original)
            combined_result["sentence"] = full_sent
            for keyvalue in simplified_models + ["ensemble_strong"]:
                combined_result[keyvalue] = [] 
                x_types, x_positions, x_prob_seqs = get_positions(converted[keyvalue][no_sample])
 #               print(x_prob_seqs)
                x_prob_seqs = [i[1] for i in converted[keyvalue][no_sample]]

                # x_prob_seqs = [(3*i[1] - 2) for i in x_prob_seqs]
                word_prob_seqs = []
                for xp in x_positions:
                    # print("seq:", xp[1], len(x_prob_seqs))
                    # if xp[1] > len(x_prob_seqs):
                        # print(keyvalue, len(x_prob_seqs), xp[1])
                    word_prob_seq = x_prob_seqs[xp[0]:xp[1]]
                    word_prob_seq = np.mean(word_prob_seq)
                    word_prob_seqs.append(word_prob_seq)
                x = fname.replace("all.json", "test.txt")
                # print("testing:x_positions", len(x_positions), len(x_types), len(word_prob_seqs)) 
                for entity_ind, (x1, x2, x3) in enumerate(zip(x_positions, x_types, word_prob_seqs)):
                    if x2 not in types: continue
                    # print("xxxx", x1,x2,x3)
                    if 'ensemble' not in keyvalue:
                        prob_strict = x3 * evaluation_metrics[x][keyvalue][x2][0]
                        prob_partial = x3 * evaluation_metrics[x][keyvalue][x2][1]
                        prob_exact = x3 * evaluation_metrics[x][keyvalue][x2][2]
                        prob_type = x3 * evaluation_metrics[x][keyvalue][x2][3]                        
                    else:

                        ## on an entity level, ensemble the metric performances of all models whose prediction yield entity within ensembled boundary
                        model_names = converted["ensemble_models"][no_sample][entity_ind]
                        c1 = []
                        c2 = []
                        c3 = []
                        c4 = []
#                        print(evaluation_metrics, x)
                        for m_name in model_names:
                            strict_pre = evaluation_metrics[x][m_name][x2][0]
                            partial_pre = evaluation_metrics[x][m_name][x2][1]
                            exact_pre = evaluation_metrics[x][m_name][x2][2]
                            type_pre = evaluation_metrics[x][m_name][x2][3]                            
                            c1.append(strict_pre)
                            c2.append(partial_pre)
                            c3.append(exact_pre)
                            c4.append(type_pre)
            #                print(x, m_name, t_name, strict_pre, partial_pre)              
                        prob_strict = x3 * (np.max(c1) + (1 - np.max(c1)) * len(c1)/(2*num_models))
                        prob_partial = x3 * (np.max(c2) + (1 - np.max(c2)) * len(c2)/(2*num_models))    
                        prob_exact = x3 * (np.max(c3) + (1 - np.max(c3)) * len(c3)/(2*num_models))
                        prob_type = x3 * (np.max(c4) + (1 - np.max(c4)) * len(c4)/(2*num_models))      
                    sent_ind = converted["ind"][no_sample]  

                    try:
                        book = data[no_sample]["book"]  
                        location = data[no_sample]["location"]  
                    except:
                        book = ''
                        location = ''
                    paragraph = data[no_sample]["paragraph"]  

                    entity1 = data[no_sample]["entity1"]    
                    source = data[no_sample]["source"]        
                    combined_result[keyvalue].append([full_sent[x1[0]:x1[1]], x1[0], x1[1], x2, prob_strict, prob_partial, prob_exact, prob_type, sent_ind])       
  #                  combined_result[keyvalue].append([full_sent[x1[0]:x1[1]], x1[0], x1[1], x2, prob_strict, prob_partial, prob_exact, prob_type, sent_ind, location, paragraph, entity1, source, book])
            combined_results.append(combined_result)

        logger.info("**后处理: finished")  
        with open("{}/{}_{}.json".format(ner_results, fname_abbrev, version_no), "w") as nrw:
            json.dump(data, nrw)

                ## 所有模型实体识别结果整合
        logger.info("*多模型预测完成，开始实体整合...")
        ## save all results for entity recognition
        reformed_results = []
        for c_r in combined_results:
            for model_name in simplified_models + ["ensemble_strong"]:
                if len(c_r[model_name]) == 0: continue
                for ent_res in c_r[model_name]:
                    reformed_results.append([ model_name] + ent_res)
        df_col = ["model", "ent_name", "start", "end", "ent_type", "prob_strict", "prob_partial", "prob_exact", "prob_type", "ind"]
        reformed_results = pd.DataFrame(reformed_results, columns = df_col)
        reformed_results["ent_cn"] = [tag_mapping[i] for i in reformed_results["ent_type"]]
        reformed_results.to_csv("{}/extraction_results_{}_{}.csv".format(ner_results, version_no, fname_abbrev))
        
if __name__ == "__main__":
    main()



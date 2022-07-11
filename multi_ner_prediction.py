"""
多语料模型分别再多个测试集评估
多集成方法的评估，包括：
1. 取各个模型预测的并集边界
2. 对各个模型根据其在各类实体的表现评估生成权重矩阵，对预测结果进行加权平均预测
3. 对个模型预测结果，首先取边界并集合并成大实体，然后取至少有两个模型支持的实体（只要模型预测的子字符串在大实体边界内就算一个）
主要功能：1. 利用测试集，根据评价指标生成各个模型评价矩阵
          2. 再测试集上对各个模型的预测进行集成，并且也进行相关的评估

使用场景：模型训练。每当模型更新后即可对各个模型进行评估生成评估的权重矩阵。
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
import yaml
import argparse
from loguru import logger
import time
# parser = argparse.ArgumentParser()


# with open("model_weights.json", "r") as f:
#     weights = json.load(f)
# sentence="5.瓣膜置换术后心内膜炎,感染严重,药物不易控制,引起人工瓣功能障碍或瓣周漏、瓣周脓肿等。需利用人工导管进行卵巢切除术，并服用阿莫西林，清咽滴丸，三位地黄片等药物并施加电热治疗。对于肠道有严重沙门氏菌感染的患者，需住院观察并输液。"
# sentence2="(1)缺血性或非缺血性心肌病(2)充分抗心力衰竭药物治疗后,NYHA心功能分级仍在Ⅲ级或不必卧床的Ⅳ级(3)窦性心律4)左心室射血分数≤35%30第2章心脏起搏(5)左心室舒张末期内径≥55mm(6)QRS波时限≥120ms伴有心脏运动不同步2.Ⅱ类适应证(1)Ⅱa类适应证①充分药物治疗后NYHA心功能分级好转至Ⅱ级,并符合Ⅰ类适应证的其他条件②慢性心房颤动患者,合乎I类适应证的其他条件,可结合房室结射频消融行CRT治疗,以保证夺获双心室(2)Ⅱb类适应证①符合常规心脏起搏适应证并心室起搏依赖的患者,合并器质性心脏病NYHA心功能Ⅲ级及以上"
# model_indicator=False
# print(sentence)


"""
多语料模型分别再多个测试集评估
"""


def predict_processing(model_name, sentence):
        ## 预测句子NER结果，结果转化为['o','B-SYM'。。。] 和概率list的两个list形式
    test1, probs_path = model_name.predict_oneline(input_str = sentence)
    results = list(test1.values())
    sentencelist = list(sentence)
    taglist = ["O" for x in sentencelist]
    for r in results:
        start_idf = 0
        for rr in np.arange(r[3][0], r[3][1]):
            if start_idf == 0:
                taglist[rr] = 'B-' + r[1]
            else:
                taglist[rr] = 'I-' + r[1]
            start_idf += 1
    return(taglist, probs_path)


def eval_saver(ner_model, types, to_save, modelidentifier):
	c_collections = []
	true_tags = []
	for hs in handled_data:
		cs, t, s = hs   
		try:
			c_results, probs_path = predict_processing(ner_model, s)
			true_tags.append(t)
			c_collections.append(c_results)
		except:
			continue
	evaluator = Evaluator(true_tags, c_collections, types)
	results_all, results_agg = evaluator.evaluate()
	destination_file = to_save + "_%s_a.json"%modelidentifier
#	print(destination_file)
	with open(destination_file, "w") as tf:
		json.dump(results_all, tf)
	destination_file = to_save + "_%s_s.json"%modelidentifier
#	print(destination_file)
	with open(destination_file, "w") as tf:
		json.dump(results_agg, tf)




def get_model_simple(modname):
    if "aug" in modname:
        clean = "_".join(modname.split("_")[-3:-1])
    else:
        clean = modname.split("_")[-2]
    return(clean)

## collect evaluation results for each entities, togethter with overall assessment.
def gen_weights(eval_folder):

    print(evaluation_path)
    files = os.listdir(eval_folder)
    samplefile_name = ""
    for f in os.listdir(eval_folder):
        if f.endswith("_s.json"):
            samplefile_name = f
            break
    with open("%s/%s"%(eval_folder,samplefile_name)) as f:
        sample = json.load(f)
    entities = list(sample.keys())
    result_agg = {}
    for e in entities:
        result_agg[e] = {}
    print("!!!!", result_agg, samplefile_name)
    results_collection = {}
    for f in files:
        if "_s.json" in f:
            result_agg = {}
            for e in entities:
                result_agg[e] = {}
            result_agg["overall"] = {}
            with open("{}/{}".format(evaluation_path, f)) as fe:
                results = json.load(fe)
                for etype, v1 in results.items():
                    for stype, scores in v1.items():
                        for metric, score in scores.items():
                            keyname = "{}_{}".format(stype, metric)
                            if metric in ["precision", "recall"]:
                                result_agg[etype][keyname] = score
            results_collection[f] = result_agg
    for f in files:
        if "_a.json" in f:
            altname = "_".join(f.split("_")[:-1])+"_s.json"
            with open("{}/{}".format(evaluation_path, f)) as fe:
                results = json.load(fe)
                for stype, scores in results.items():
                    for metric, score in scores.items():
                        keyname = "{}_{}".format(stype, metric)
                        if metric in ["precision", "recall"]:
                            results_collection[altname]["overall"][keyname] = score
    weightdic = defaultdict(list)
    model_names= []
    for e,v in results_collection.items():
        if "seed" not in e: continue
        if "aug" in e: continue
        model_names.append(get_model_simple(e))
        for k,vv in v.items():
            weightdic[k].append(vv["strict_precision"])
    weightdic2 ={}
    for k,v in weightdic.items():
        sumweight = np.sum(v)
        weights = {}
        for m, w in zip(model_names ,v):
            weights[m] = (w/sumweight)
        weightdic2[k] = weights
    weightdic3 = defaultdict(dict)
    for k,v in weightdic2.items():
        for kk,vv in v.items():
            weightdic3[kk][k] = vv 
    with open ("{}/model_weights.json".format(ner_results), "w") as f:
        json.dump(weightdic3, f)


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
        #print(removal_list)
        for model, slist in dic.items():
            for rl in removal_list:
                slist.pop(rl)
            dic[model] = slist
        return(dic)



        
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


def main():
    global handled_data, evaluation_path, ner_results
    with open('config.yaml', 'rb') as fp:
        dic_file = yaml.safe_load(fp)
        types =  list(dic_file['ontologies']['RE'].keys())
        model_path = dic_file['model_path']
        desirable_models = dic_file['prediction_model']
        ner_training_final = dic_file['training_data']['ner_training_final']
        evaluation_path = dic_file['evaluation_path']
        ner_results = dic_file['results']['ner_results']

    if not os.path.exists(evaluation_path):
        os.makedirs(evaluation_path)
    if not os.path.exists(ner_results):
        os.makedirs(ner_results)
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path =  'logs/seqlabeling_'
    logfile = log_path + rq + '.log'
    logger.add(logfile, backtrace=True, diagnose=True, rotation='3 days', retention='2 months')
    logger.info("The model evaluation path is created!")
    """
#     1. 首先生成各个模型对测试数据的预测结果评估，
#     每个模型对每个数据集的评估结果分别并储存在evaluation文件夹下的对应文件中
#     """
    multi_models = {}

    for model_name in os.listdir(model_path):
        if model_name in desirable_models:
#             print("!!!!!!!!!!!!loading model: {}/{}".format(model_path, model_name))
            ner_model = ChineseNER("predictonly", "{}/{}".format(model_path, model_name))
            multi_models[get_model(model_name)] = ner_model

    test_datasets = [x for x in os.listdir(ner_training_final) if "_test" in x]
    ## test datasets 是我们感兴趣评估的测试集，可以选择使用每一类类型数据集的测试集分别评估，这里默认只用全量混合数据测试集

    logger.info("step 1: models loaded,{},test datasets are: {}".format(multi_models, " ".join(test_datasets)))

    for x in test_datasets:
        to_save = evaluation_path + "/" + x.split(".")[0]
        data_path = "{}/{}".format(ner_training_final, x)
        sentence = ''
        target = []
        handled_data = []
        char_sentence = []
        line_cnt = 0
        sent_cnt = 0
        for line in codecs.open(data_path, 'r', 'utf8'):
            line_cnt += 1
            line = line.strip()
            if line == "O": continue
            if line == "" or line == "\t" or line == " " or line == "\n":    
                # 已经收集好一条句子了，加入到seq_list中
                sent_cnt += 1
                handled_data.append([char_sentence, target, sentence])
                #print("conditions:",self.data[-1][-2:])
                sentence = ''
                target = []
                char_sentence = []
                continue
            else:
                try:
                    word, tag = line.split(" ")
                except Exception:
                    try:
                        word, tag = line.split("\t")
                    except Exception:
                        continue
                # 将当前的字符和tag加入seq中（seq本身也是list）
                if word == "":
                    print("no", line)
                char_sentence.append(word)
                target.append(tag)
                sentence += word
        for ner_mod_name, ner_mod_itself in multi_models.items():
            logger.info("step 2: model: {} evaluation completed.".format(ner_mod_name))
            eval_saver(ner_mod_itself, types, to_save, ner_mod_name)

    """2. Then generate weight matrix combining all evaluation results for each model/entity combo for later use"""
    gen_weights(evaluation_path)
    logger.info("step 3: weights matrix saved")
    with open("{}/model_weights.json".format(ner_results), "r") as f:
        weights = json.load(f)
    """
    3. 对每个测试数据集分别预处理，并且各种集成处理
    """

    results_full = []
    evaluation_metrics = {}
    combined_results = []
    simplified_models = [get_model(m) for m in desirable_models]
    for x in test_datasets:
        evaluation_metrics[x] = {}
#        if "seed" not in x: continue
        to_save = "{}/".format(evaluation_path) + x.split(".")[0]
        data_path = "{}/{}".format(ner_training_final, x)
        sentence = ''
        target = []
        handled_data = []
        char_sentence = []
        line_cnt = 0
        sent_cnt = 0

        for line in codecs.open(data_path, 'r', 'utf8'):
            line_cnt += 1
            line = line.strip()
            if line == "O": continue
            if line == "" or line == "\t" or line == " " or line == "\n":    
                # 已经收集好一条句子了，加入到seq_list中
                sent_cnt += 1
                handled_data.append([char_sentence, target, sentence])
                #print("conditions:",self.data[-1][-2:])
                sentence = ''
                target = []
                char_sentence = []
                continue
            else:
                try:
                    word, tag = line.split(" ")
                except Exception:
                    try:
                        word, tag = line.split("\t")
                    except Exception:
                        continue
                # 将当前的字符和tag加入seq中（seq本身也是list）
                if word == "":
                    print("no", line)
                char_sentence.append(word)
                target.append(tag)
                sentence += word
        c_collections = []
        logger.info("step 4: test data prepared")
        for idx, hs in enumerate(handled_data):
#            if idx >= 100: break
            cs, t, s = hs   
            pred_dict = {}

            ## get the model predctions and convert them into the [(tag, prob),...] sequence format
            for pa, mod in multi_models.items():
                try:
                    c_results, probs_path = predict_processing(mod, s)
#                    print(c_results, probs_path)
                except:
                    c_results = []
                
                label_prob = [(i1, i2) for i1,i2 in zip(c_results, probs_path)]
                pred_dict[pa] = label_prob
                #print(pred_dict)

                #pred_dict["probs_path"] = probs_path
            pred_dict["original"] = t
            label_ensembles = {}

            """3.1 交集集成处理"""
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

            """3.2 并集集成法"""
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

            """3.3 并集后交集投票集成法：先并集 后根据数量筛选(并集的ent type recall 
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
                    
            ## 并集后交集投票集成法：形式复原
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
            c_collections.append(pred_dict)
        logger.info("step 5: prediction and ensembling completed")

        # 存储： 集成结果的字典列表，并对字典列表变形
        with open("{}/evaluation_ensemble.json".format(ner_results), "w") as ff:
            json.dump(c_collections, ff)
        converted = defaultdict(list)
        for v in c_collections:
            for m,l in v.items():
                converted[m].append(l)
        converted = result_cleaning(converted)
        
        """
        4. 对各种模型，包括集成模型的结果进行评估（MUR方法）并保存到evaluation_metrics中

        """

        for keyvalue in simplified_models + ["ensemble_strong"]:
    #        print(len(converted["original"]), len(converted[keyvalue]))
            labels_only = []
            for ck in converted[keyvalue]:
                labels_only_sub = []
                for ck_sub in ck:
                    labels_only_sub.append(ck_sub[0])
                labels_only.append(labels_only_sub)
            ## evaluate and append results to evaluation metric
            evaluator = Evaluator(converted["original"], labels_only, types)
            results_all, results_agg = evaluator.evaluate()
            evaluation_metrics[x][keyvalue] = {}
            for e_type_eval, eval_dic in results_agg.items():
                evaluation_metrics[x][keyvalue][e_type_eval] = [eval_dic["strict"]["precision"], eval_dic["partial"]["precision"], \
                    eval_dic["exact"]["precision"], eval_dic["ent_type"]["precision"]]
                results_full.append([x,  keyvalue, e_type_eval, results_agg[e_type_eval]["partial"]["precision"], results_agg[e_type_eval]["partial"]["recall"], \
                results_agg[e_type_eval]["exact"]["precision"], results_agg[e_type_eval]["exact"]["recall"], results_agg[e_type_eval]["ent_type"]["precision"], \
                    results_agg[e_type_eval]["ent_type"]["recall"],results_agg[e_type_eval]["strict"]["precision"], results_agg[e_type_eval]["strict"]["recall"]])
            results_full.append([x,  keyvalue, "all_types", results_agg[e_type_eval]["partial"]["precision"], results_agg[e_type_eval]["partial"]["recall"], \
                results_agg[e_type_eval]["exact"]["precision"], results_agg[e_type_eval]["exact"]["recall"], results_agg[e_type_eval]["ent_type"]["precision"], \
                    results_agg[e_type_eval]["ent_type"]["recall"],results_agg[e_type_eval]["strict"]["precision"], results_agg[e_type_eval]["strict"]["recall"]])
    logger.info("step 6: evaluation completed")
    
    with open("{}/eval_dic.json".format(evaluation_path), "w") as jsoneval:
        json.dump(evaluation_metrics ,jsoneval)
        
if __name__ == "__main__":
    main()

#
#
#
#

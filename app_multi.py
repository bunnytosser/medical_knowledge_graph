
from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle as p
import json
from collections import defaultdict

app = Flask(__name__)

import numpy as np
import os
import re
import json
import jieba
import jieba.posseg
from ner.multi_source_ner import ChineseNER
tag_mapping = {
            "DIS": "疾病",
            "SYM": "症状",
            # "SGN": "体征",
            # "REG": "部位词",
            "ORG": "器官",
            "BFL": "体液",
            "DEG": "程度词",
            "FW": "频率词",
            "DUR": "时间段",
            "TP": "时间点",
            "TES": "检查",
            "DRU": "药品",
            "SUR": "手术",
            "PRE": "措施(非手术、非药品)",
            # "PRP": "干预(治疗原则)",
            "CL": "条件词",
            "PSB": "可能性词",
            "PT": "既往信息词",
            "AT": "否认词",
            "O": "非关键词",
            "NBP": "待处理",
            "BRS": "血亲情况",
            "SPS": "配偶状况",
            "BAC": "微生物",
            "MAT": "耗材",
            "OBJ": "物质",
        }
def convert_to_seq(jsn, pseq):
    pseq = [["O", j] for j in pseq]
    for j in jsn.values():

        locs = j[3]
        ids = 0
        for l in range(locs[0], locs[1]):
            if ids == 0:
                pseq[l][0] = "B-%s"%j[1]
            else:
                pseq[l][0] = "I-%s"%j[1]
            ids += 1
    return(pseq)
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
#sentence="5.瓣膜置换术后心内膜炎,感染严重,药物不易控制,引起人工瓣功能障碍或瓣周漏、瓣周脓肿等"
sentence2="(1)缺血性或非缺血性心肌病(2)充分抗心力衰竭药物治疗后,NYHA心功能分级仍在Ⅲ级或不必卧床的Ⅳ级(3)窦性心律4)左心室射血分数≤35%30第2章心脏起搏(5)左心室舒张末期内径≥55mm(6)QRS波时限≥120ms伴有心脏运动不同步2.Ⅱ类适应证(1)Ⅱa类适应证①充分药物治疗后NYHA心功能分级好转至Ⅱ级,并符合Ⅰ类适应证的其他条件②慢性心房颤动患者,合乎I类适应证的其他条件,可结合房室结射频消融行CRT治疗,以保证夺获双心室(2)Ⅱb类适应证①符合常规心脏起搏适应证并心室起搏依赖的患者,合并器质性心脏病NYHA心功能Ⅲ级及以上"
model_indicator=False
def entitylen(dic):
	len1=[len(i[0]) for i in dic.values()]
	avelen=np.mean(len1)
	maxlen=np.max(len1)
	print(maxlen,avelen)
	if maxlen>=5 and avelen>=3:
		return True
s1="adverse"
s2="indications"
#test1 = cn.predict_oneline(sentence,s1)

@app.route('/', methods=['POST','GET'])
def makeprediction():
    all_res = defaultdict()
    #preds = model.predict_oneline(sentence2)
    data = request.get_json(force=True)
    sentence = data['sentence']
    for mname, model in multi_models.items():
        prediction = model.predict_oneline(sentence)[0]
        sequences = model.predict_oneline(sentence)[1]
        all_res[mname] = defaultdict()
        all_res[mname]["preds"] = prediction
        all_res[mname]["probs"] = sequences
    converted = {}
    for k in all_res.keys():
        res = convert_to_seq(all_res[k]["preds"], all_res[k]["probs"])
        converted[k] = res
    ensemble_union = []
    prob_union = []

    for (xi, xi_p) in converted["a"]:
        prob_union.append(xi_p)
        if xi == 'O': 
            ensemble_union.append(xi)
        else:
            etype = xi.split("-")[1]
            ensemble_union.append(etype)

    for model, lseq in converted.items():
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
    ensemble_majority = {}
    for no in range(len(ensemble_union)):
        ensemble_majority[no] = defaultdict(list)
    # 构建一个记录并集标签后 词语起始位置的列表行字典，方便后续往每个合并后词 append 模型结果
    # recorder = {}
    positions = []
    len_seq = len(ensemble_union) - 1
    init_pos = [0, 0]
    waiting = False
    c_types = [] 
    c_types, positions, prob_seqs = get_positions(ensemble_union)
    #           print("prob seqs", prob_seqs)

    frequency_counts = defaultdict(list)
    prob_combined = [i for (j,i) in converted["a"]]
    for model, lseq in converted.items():
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
    # 存储： 集成结果的字典列表，并对字典列表变形
    preds = {}
    i = 0
    c_types, positions, prob_seqs =  get_positions(voted_seq)
    for c, p, ps in zip(c_types, positions, prob_seqs):
        preds[i] = [sentence[p[0]: p[1]+1], c, tag_mapping[c], p, ps]
        i += 1
    return(jsonify(preds))

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

if __name__ == '__main__':
    desirable_models = [
    #'params_o_cnn_m.pkl',
    'params_o_cnn_d.pkl',
    'params_o_cnn_c.pkl',
    'params_o_cnn_k.pkl',
    'params_o_cnn_a.pkl',
    'params_o_cnn_m_v2.pkl',
    'params_o_cnn_a_v2.pkl']
    multi_models = {}
    for model_name in os.listdir("models"):
        if model_name in desirable_models:
            multi_models[get_model(model_name)] = ChineseNER("predictonly", "models/" + model_name)


    app.run(host='192.168.4.34', port = 4111 ,debug=True)

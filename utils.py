import os
import re
import requests
import numpy as np
import pandas as pd
import ast
import json
import yaml
from multi_source_ner import ChineseNER
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
import copy

class Model_selector:
    
    """choose the model that gives more entities and average entity length"""

    def __init__(self):
        self.sentence = "5.瓣膜置换术后心内膜炎,感染严重,药物不易控制,引起人工瓣功能障碍或瓣周漏、瓣周脓肿等。此时应使用用纱布，血压计，心电图仪对病患进行监测。"
        self.sentence2 = "(1)缺血性或非缺血性心肌病(2)充分抗心力衰竭药物治疗后,NYHA心功能分级仍在Ⅲ级或不必卧床的Ⅳ级(3)窦性心律4)左心室射血分数≤35%30第2章心脏起搏(5)左心室舒张末期内径≥55mm(6)QRS波时限≥120ms伴有心脏运动不同步2.Ⅱ类适应证(1)Ⅱa类适应证①充分药物治疗后NYHA心功能分级好转至Ⅱ级,并符合Ⅰ类适应证的其他条件②慢性心房颤动患者,合乎I类适应证的其他条件,可结合房室结射频消融行CRT治疗,以保证夺获双心室(2)Ⅱb类适应证①符合常规心脏起搏适应证并心室起搏依赖的患者,合并器质性心脏病NYHA心功能Ⅲ级及以上"
        self.model_indicator = False
    def entitylen(self, dic):
        len1 = [len(i[0]) for i in dic.values()]
        avelen = np.mean(len1)
        maxlen = np.max(len1)
        print(maxlen,avelen)
        if maxlen >= 2 and avelen >= 2:
            return True
    def selection(self, model_path):
        while self.model_indicator == False:
            cn = ChineseNER("predictonly",model_path)
            test1 = cn.predict_oneline(input_str = self.sentence)[0]
            test2 = cn.predict_oneline(input_str = self.sentence2)[0]
            print("still searching",len(test1),len(test2))
            if len(test1) >= 2 and len(test2) >= 2:
                if self.entitylen(test1) and self.entitylen(test2):
                    self.model_indicator = True
                    print("ideal model found!",test1,test2)
        return(cn)


class Utillist:
    def __init__(self):
        with open('config.yaml', 'rb') as fp:
            dic_file = yaml.safe_load(fp)
            stoplist = dic_file['dictionaries']['stop']
            self.fulldic = dic_file['dictionaries']['full']
        self.exclusions = []
        with open(stoplist,"r") as f:
            for line in f:
                self.exclusions.append(line.strip())
    def fetch_dic(self, stype):
        """get the latest version of dicitionary"""
        with open(self.fulldic) as f:
            dics = json.load(f)
        com = []
        for key,value in dics.items():
            if value == stype:
                com.append(key)
        com.sort(key=len)
        com = com[::-1]
        return(com)

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 4, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
class prefix_tool():
    def __init__(self,wordlist):
        self.wordlist=wordlist
        self.dictionary_words_cut=[]
        self.dictionary_flags_cut=[]
        self.__init_run__()
        self.__prefix_finder__()
        exclusions = []

    def __init_run__(self):
        for j in self.wordlist:
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

    def __prefix_finder__(self):
        suffixdic = defaultdict(int)
        prefixdic = defaultdict(int)
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
            






# class Exploration():
#     def __init__(self, indications, focusedtype,suffix=["sy"]):
#         self.tags=[]
#         self.words=[]
#         self.sentences=[]
#         self.suffix=dissuf
#         self.kwposition=[]
#         self.entity1=[]
#         self.indexes=[]
#         self.ind_sent={}
#         self.exl_tags = ["u","m","x","c","p","r","d","v"]  
#         num_word=0
#         for ix, j in enumerate(indications):
#             sent=j["sentence"]
#             segs=j["seg"]
#             dic_ind=j["ind"]
#             try:
#                 segstype=[s[1] for s in segs]
#                 segsword=[s[0] for s in segs]
#             except:
#                 print(print(j))
#                 print(segs[0])
#                 print(segs[0][1])
#                 print()
#             for m,ss in enumerate(segstype):
#                 if ss==focusedtype:
#                     mmax=np.min((m+5,len(segstype)))
#                     mmin=np.max((m-5,0))
#                     self.tags.append(segstype[mmin:mmax])
#                     self.words.append(segsword[mmin:mmax])
#                     self.kwposition.append(m-mmin)
#                     self.sentences.append(sent)
#                     self.entity1.append(j["entity1"])
#                     self.indexes.append(dic_ind)
#                     num_word+=1
#         for (p,q) in zip(self.indexes,self.sentences):
#             self.ind_sent[p]=q
#         print(num_word)
#         self.reorgnized_w,self.reorgnized_t,self.reorgnized_i=results=self.__processing__(focusedtype)
# #         self.prew=
# #         self.pret= 
#     def __processing__(self,focusedtype):
#         all_t=[]
#         all_w=[]
#         quadruple_dic={}
#         for i,t in enumerate(self.tags):
#             try:
#                 focal=t.index(focusedtype)
#             except:
#                 print(t)
#             if focal==0:
#                 continue
#             for n in range(-2,3,1):
#                 if n==0:
#                     continue
#                 # appending lists of words/tags fragments to the mega dict repectively
#                 elif n<0 and focal+n!=0:
#                     start=np.max((0,focal+n))
#                     end=focal+1
#                     sliced=t[start:end]
#                 elif n>0 and focal+n!=len(t):
#                     start=focal
#                     end=np.min((len(t),focal+n))+1
#                     sliced=t[start:end]
#                 else:
#                     continue
#                 conword=" ".join(self.words[i][start:end])
#                 w_key="w "+str(n+2)+"|"+str(self.indexes[i])
#                 t_key="t "+str(n+2)+"|"+str(self.indexes[i])
#                 quadruple_dic[w_key]=conword
#                 quadruple_dic[t_key]=sliced   
#         self.quadruple_dic=quadruple_dic       
#         reorgnized_w=defaultdict(list)
#         reorgnized_t=defaultdict(list)
#         reorgnized_i=defaultdict(list)
#         for k,v in quadruple_dic.items():
#             order=int(k[2])
#             ind=int(k.split("|")[1])
#             if k[0]=="w":
#                 reorgnized_w[order].append(v) 
#                 reorgnized_i[order].append(ind)
#             else:
#                 reorgnized_t[order].append(v) 
#         ## 返回两个字典，分别是词 和 词性。不同key表示中心词前后不同位置。
#         return(reorgnized_w,reorgnized_t,reorgnized_i)

#     def plot(self, pos):
#         #展示特定位置的词的词性构成
# #         pie, ax = plt.subplots(figsize=[6,6])
#         s=pos+2
#         typefrequency=defaultdict(int)
#         if s-2<0:
#             subdict=[i[0] for i in self.reorgnized_t[s]]
#         else:
#             subdict=[i[-1] for i in self.reorgnized_t[s]]
#         for j in subdict:
#             typefrequency[j]+=1
#         labels = []
#         sizes = []
#         for x, y in typefrequency.items():
#             labels.append(x)
#             sizes.append(y)
# #         plt.pie(sizes, labels=labels, pctdistance=0.5)
#         sns.barplot(x=labels,y=sizes)
# #     def nested_plot(self, pos):
        
#     def type_details(self,specifictype, pos):
#         # specifictype:想看的位置为pos的分词类型为specifictype的词
#         results=[]
#         if pos<0:
#             specific_t=[i[0] for i in self.reorgnized_t[pos+2]]
#         else:
#             specific_t=[i[-1] for i in self.reorgnized_t[pos+2]]
#         n_w=self.reorgnized_w[pos+2] 
#         n_t=self.reorgnized_t[pos+2] 
        
#         for i,(t,w) in enumerate(zip(specific_t,n_w)):

#             if t==specifictype and not any(tag in n_t[i] for tag in self.exl_tags):
#                 sentence_index=self.reorgnized_i[pos+2][i]
#                 sentence=self.ind_sent[sentence_index]
#                 new_entity=w.replace(" ","")
#                 findind=sentence.find(new_entity)
#                 display_sent=sentence[:findind]+"【"+new_entity+"】"+sentence[findind+len(new_entity):]
                
#                 result={}
#                 result["words"]=w
#                 result["tags"]=n_t[i]
#                 result["sentence"]=display_sent
#                 results.append(result)
#         return(results)
    
#     def is_a_in_x(self, A, X):
#         for i in range(len(X) - len(A) + 1):
#             if A == X[i:i+len(A)]: return (i,True)
#         return (0,False)
    
#     def seq_examiner(self,seq):
#         for j in indications:
#             segs=[s[1] for s in j["seg"]]
#             words=[s[0] for s in j["seg"]]
#             (pos,contains)=self.is_a_in_x(seq,segs)
#             if contains: 
#                 print(words[pos-1:pos+4])
                
# #     def seq_examiner_fuzzy(self,seq):
# #         for s in seq:
# #             if s=="":
# #                 for seg in 
# #         for j in indications:
# #             segs=[s[1] for s in j["seg"]]
# #             words=[s[0] for s in j["seg"]]
# #             (pos,contains)=self.is_a_in_x(seq,segs)
# #             if contains: 
# #                 print(words[pos-1:pos+4])
    
#     def suffix_combiner(self,full_display=False):
#         for i,j in enumerate(self.words):
#             m=self.kwposition[i]
#             if len(j)<=1:
#                 continue
#             suffix_included=[i for i in j[m:m+2] if i in self.suffix]
#             if suffix_included!=[]:
#                 first_suffix=suffix_included[0]
#                 until=j[m:].index(first_suffix)+m
# #                 print(j[m],self.tags[i][m],m,until)
#                 if len(j[m:until+1])>1 and not any(tag in self.tags[i][m:until+1] for tag in self.exl_tags):
#                     sent=self.sentences[i]
#                     display_front=np.max((0,m-5))
#                     display_end=np.min((until+6,len(j)))
#                     new_entity="".join(j[m:until+1])
#                     findind=sent.find(new_entity)
                    
#                     display_sent=sent[:findind]+"【"+new_entity+"】"+sent[findind+len(new_entity):]
#                     wlist=self.sentences[i]
#                     ent1=self.entity1[i]
# #                     sent="".join(wlist[:m])+"【"+"".join(wlist[m:until+1])+"】"+"".join(wlist[until+1:])
#                     #### 某系类型的分词词性 数字不能出现在短语中
#                     if m==0:
#                         continue 
#                     if self.tags[i][m-1] in ["og","sy","ds","bc","bf"]:
                        
# #                         start=j["seg"][pos-1][2][1]
# #                         combined="".join(words[pos:pos+len(seq)])
# #                         if "、" in combined:
# #                             continue
# #                         newly_merged[seqkey].append(combined)
# #                         j["seg"][pos:pos+len(seq)]=[[combined,seq[-1],[start,start+len(combined)]]]
                    
#                         print(ent1,j[m-1:until+1],self.tags[i][m-1:until+1],display_sent)
#                     else:
#                         print(ent1,j[m:until+1],self.tags[i][m:until+1],display_sent)
#                     if full_display==True:
#                         print(j,"\n",self.tags[i])
#                     print()
                    
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
            print(j["seg"])
        for seq in po_series:
            # iterating through all candidate sequences
            seqkey = "|".join(seq)
    #         newly_merged[seqkey].append("h")
            (pos,contains) = is_a_in_x(seq,segs)
            if contains: 
                try:
                    start = j["seg"][pos-1][2][1]
                    combined = "".join(words[pos:pos+len(seq)])
                    if "、" in combined or combined not in j["sentence"] or "," in combined or ":" in combined:
                        continue
                    print(combined,seq)
                    newly_merged[seqkey].append(combined)
#                     j["seg"][pos:pos+len(seq)]=[[combined,seq[-1],[start,start+len(combined)]]]
                    j_dict[pos] = [combined,seq[-1],[start,start+len(combined)]]
                    del j_dict[pos+1]
                    if len(seq) == 3:
                        del j_dict[pos+2]
                except:
                    print(j["seg"])
        j["seg"] = list(j_dict.values())
    lennew = 0
    for j in newly_merged.values():
        lennew += len(j)
    print("newly discovered combinations:",lennew) 
    return(indications_copy)
#         newly_merged.append(new_words)

## 合并符合在指定位置前2个词内出现指定词性/实体的词，就进行合并并且返回更新的全量数据
def words_combiner_fuzzy(merged_results,pre_types=["sy","og"],centertype="pr"):
# po_series=set(po_series)
    po_series = []
    for pt in pre_types:
        po_series.append([pt,centertype])
        for t in all_tags:
            if t in ["u","m","x","c","p","r","d","v"]  :
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
            print(j["seg"])
        for seq in po_series:
            # iterating through all candidate sequences
            seqkey = "|".join(seq)
    #         newly_merged[seqkey].append("h")
            (pos,contains) = is_a_in_x(seq,segs)
            if contains: 
                try:
                    start = j["seg"][pos-1][2][1]
                    combined = "".join(words[pos:pos+len(seq)])
                    if "、" in combined or combined not in j["sentence"] or "," in combined or ":" in combined:
                        continue

                    newly_merged[seqkey].append(combined)
#                     j["seg"][pos:pos+len(seq)]=[[combined,seq[-1],[start,start+len(combined)]]]
                    j_dict[pos] = [combined,seq[-1],[start,start+len(combined)]]
                    print(combined,seq)
                    del j_dict[pos + 1]
                    if len(seq) == 3:
                        del j_dict[pos+2]
                except:
                    print(j["seg"])
        j["seg"] = list(j_dict.values())
    lennew = 0
    for j in newly_merged.values():
        lennew += len(j)
    print("newly discovered combinations:",lennew) 
    return(indications_copy)
#         newly_merged.append(new_words)

## 合并后缀
def suffix_combiner(merged_results,suffix,desiredlist = ["og","sy","ds","bc","bf"],suffixtype="ds"):
# po_series=set(po_series)
    indications_copy = copy.deepcopy(merged_results)
    newly_merged = defaultdict(list)
    for j in indications_copy:
        j_dict={}
        
        for z1,z2 in zip(list(range(0,len(j["seg"]))),j["seg"]):
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
                j_dict[pos] = [combined,suffixtype,[start,start+len(combined)]]
                print(combined)
                del j_dict[pos+1]
#                     j["seg"][pos:pos+len(seq)]=[[combined,seq[-1],[start,start+len(combined)]]]
    #             j["seg"[pos:pos+len(seq)]=["".join(words[pos:pos+len(seq)]),seq[-1]]
        j["seg"] = list(j_dict.values())
#     print(j["seg"])
    lennew = 0
    for j in newly_merged.values():
        lennew+=len(j)
    print("newly discovered combinations:",lennew) 
    return(indications_copy)
#         newly_merged.append(new_words)

class Exploration():
    def __init__(self, indications, focusedtype, suffix=["SYM"]):
        self.tags = []
        self.words = []
        self.sentences = []
        self.suffix = dissuf
        self.kwposition = []
        self.entity1 = []
        self.indexes = []
        self.ind_sent = {}
        num_word = 0
        for ix, j in enumerate(indications):
            sent = j["sentence"]
            segs = j["seg"]
            dic_ind = j["ind"]
            try:
                segstype = [s[1] for s in segs]
                segsword = [s[0] for s in segs]
            except:
                print(print(j))
                print(segs[0])
                print(segs[0][1])
                print()
            for m,ss in enumerate(segstype):
                if ss == focusedtype:
                    mmax = np.min((m + 5,len(segstype)))
                    mmin = np.max((m - 5,0))
                    self.tags.append(segstype[mmin:mmax])
                    self.words.append(segsword[mmin:mmax])
                    self.kwposition.append(m-mmin)
                    self.sentences.append(sent)
                    self.entity1.append(j["entity1"])
                    self.indexes.append(dic_ind)
                    num_word += 1
        for (p,q) in zip(self.indexes,self.sentences):
            self.ind_sent[p] = q
        print(num_word)
        self.reorgnized_w,self.reorgnized_t,self.reorgnized_i=results=self.__processing__(focusedtype)
#         self.prew=
#         self.pret= 
    def __processing__(self,focusedtype):
        all_t = []
        all_w = []
        quadruple_dic = {}
        for i,t in enumerate(self.tags):
            try:
                focal=t.index(focusedtype)
            except:
                print(t)
            if focal == 0:
                continue
            for n in range(-2,3,1):
                if n == 0:
                    continue
                # appending lists of words/tags fragments to the mega dict repectively
                elif n < 0 and focal + n != 0:
                    start = np.max((0,focal+n))
                    end = focal+1
                    sliced = t[start:end]
                elif n > 0 and focal + n != len(t):
                    start = focal
                    end = np.min((len(t), focal + n)) + 1
                    sliced = t[start:end]
                else:
                    continue
                conword = " ".join(self.words[i][start:end])
                w_key = "w " + str(n+2)+"|"+str(self.indexes[i])
                t_key = "t "+str(n+2)+"|"+str(self.indexes[i])
                quadruple_dic[w_key]=conword
                quadruple_dic[t_key]=sliced   
        self.quadruple_dic = quadruple_dic       
        reorgnized_w = defaultdict(list)
        reorgnized_t = defaultdict(list)
        reorgnized_i = defaultdict(list)
        for k,v in quadruple_dic.items():
            order = int(k[2])
            ind = int(k.split("|")[1])
            if k[0] == "w":
                reorgnized_w[order].append(v) 
                reorgnized_i[order].append(ind)
            else:
                reorgnized_t[order].append(v) 
        ## 返回两个字典，分别是词 和 词性。不同key表示中心词前后不同位置。
        return(reorgnized_w,reorgnized_t,reorgnized_i)

        
    def type_details(self,specifictype, pos):
        # specifictype:想看的位置为pos的分词类型为specifictype的词
        results = []
        if pos < 0:
            specific_t = [i[0] for i in self.reorgnized_t[pos+2]]
        else:
            specific_t = [i[-1] for i in self.reorgnized_t[pos+2]]
        n_w = self.reorgnized_w[pos+2] 
        n_t = self.reorgnized_t[pos+2] 
        
        for i,(t,w) in enumerate(zip(specific_t,n_w)):

            if t == specifictype and not any(tag in n_t[i] for tag in exl_tags):
                sentence_index = self.reorgnized_i[pos+2][i]
                sentence = self.ind_sent[sentence_index]
                new_entity = w.replace(" ","")
                findind = sentence.find(new_entity)
                display_sent = sentence[:findind]+"【"+new_entity+"】"+sentence[findind+len(new_entity):]
                
                result={}
                result["words"] = w
                result["tags"] = n_t[i]
                result["sentence"] = display_sent
                results.append(result)
        return(results)
    
    def is_a_in_x(self, A, X):
        for i in range(len(X) - len(A) + 1):
            if A == X[i:i+len(A)]: return (i,True)
        return (0,False)
    
    def seq_examiner(self,seq):
        for j in indications:
            segs=[s[1] for s in j["seg"]]
            words=[s[0] for s in j["seg"]]
            (pos,contains)=self.is_a_in_x(seq,segs)
            if contains: 
                print(words[pos-1:pos+4])
                
#     def seq_examiner_fuzzy(self,seq):
#         for s in seq:
#             if s=="":
#                 for seg in 
#         for j in indications:
#             segs=[s[1] for s in j["seg"]]
#             words=[s[0] for s in j["seg"]]
#             (pos,contains)=self.is_a_in_x(seq,segs)
#             if contains: 
#                 print(words[pos-1:pos+4])
    
    def suffix_combiner(self,full_display=False):
        for i,j in enumerate(self.words):
            m=self.kwposition[i]
            if len(j) <= 1:
                continue
            suffix_included = [i for i in j[m:m+2] if i in self.suffix]
            if suffix_included != []:
                first_suffix = suffix_included[0]
                until = j[m:].index(first_suffix)+m
#                 print(j[m],self.tags[i][m],m,until)
                if len(j[m:until+1]) > 1 and not any(tag in self.tags[i][m:until+1] for tag in exl_tags):
                    sent = self.sentences[i]
                    display_front = np.max((0,m-5))
                    display_end = np.min((until+6,len(j)))
                    new_entity = "".join(j[m:until+1])
                    findind=sent.find(new_entity)
                    
                    display_sent = sent[:findind]+"【"+new_entity+"】"+sent[findind+len(new_entity):]
                    wlist = self.sentences[i]
                    ent1 = self.entity1[i]
#                     sent="".join(wlist[:m])+"【"+"".join(wlist[m:until+1])+"】"+"".join(wlist[until+1:])
                    #### 某系类型的分词词性 数字不能出现在短语中
                    if m == 0:
                        continue 
                    if self.tags[i][m-1] in ["og","sy","ds","bc","bf"]:
                        
#                         start=j["seg"][pos-1][2][1]
#                         combined="".join(words[pos:pos+len(seq)])
#                         if "、" in combined:
#                             continue
#                         newly_merged[seqkey].append(combined)
#                         j["seg"][pos:pos+len(seq)]=[[combined,seq[-1],[start,start+len(combined)]]]
                    
                        print(ent1, j[m-1:until+1], self.tags[i][m-1:until+1], display_sent)
                    else:
                        print(ent1, j[m:until+1], self.tags[i][m:until+1], display_sent)
                    if full_display == True:
                        print(j, "\n", self.tags[i])
                    print()


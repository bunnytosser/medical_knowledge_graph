import os
import re
import requests
import numpy as np
import pandas as pd
import ast
import json
from utils import prefix_tool, Utillist
import jieba.posseg
from collections import defaultdict
import requests
import yaml

with open('config.yaml', 'rb') as fp:
    dic_file = yaml.safe_load(fp)
    clinicals = dic_file['corpus']['clinicals']
    manuals = dic_file['corpus']['manuals']
    preprocessed_manual = dic_file['data_files']['preprocessed_manual']
    preprocessed_clinical = dic_file['data_files']['preprocessed_clinical']
utillist = Utillist()
dis = utillist.fetch_dic(stype="DIS")
exclusions = utillist.exclusions
falsealarm = ["检查","表现","治疗","并发症","疼痛","死亡","检査","手术","临床","分类","。",";"]
def get_cutting(cutting_line,tlen):
    cutters = []
    for i,l in enumerate(cutting_line):
        if i == len(cutting_line)-1:
            cutters.append((l,tlen))
        else:
            cutters.append((l,cutting_line[i+1]))
    return cutters

"""return the matched disease name given a string. Different types of regex matching are combined,
including matching by keyword for disease title identification; identifying the "、XXX疾病" pattern;
identifying "三) XXX" or "三、XXX" pattern. Finally if no regex pattern is found, it would search through 
the entire disease list for potential matches
"""

def get_disease2(j):
    disease = None
    pattern = re.compile('^\u7b2c.{1,3}\u90e8\u5206|^\u7b2c.{1,3}\u7ae0.*|^\u7b2c.{1,3}\u8282|^\u7b2c.{1,3}\u7bc7.*')
    result = re.match(pattern,j)
    mres = re.match(".*\u4e00、|.*\u4e8c、|.*\u4e09、|.*\u56db、|.*\u4e94、|.*\u516d、\
    |.*\u4e03、|.*\u516b、|.*\u4e5d、|.*\u5341、",j)
    nres=re.match(".*\u4e00\)|.*\u4e8c\)|.*\u4e09\)|.*\u56db\)|.*\u4e94\)\
    |.*\u516d\)|.*\u4e03\)|.*\u516b\)|.*\u4e5d\)|.*\u5341\)",j)
    if result is not None:
        if "篇" in j:
            disease = j.split("篇",1)[1]
        elif '章'in j:
            disease = j.split("章",1)[1]
        elif "节" in j:
            disease = j.split("节",1)[1]
    elif mres is not None:
#             print(mres)
        disease = j.split("、")[1]
    elif nres is not None:
#             print(nres)
        disease = j.split(")")[1]
    else:
        for it in dis:
            if it in j:
                disease = it
                break
    if disease in exclusions:
        disease = None
    return disease

def change(disease,fl,book_name):
    ## Initializing disease content dictionary when a new disease name line is found 
    dic={}
    indicator = 0
    checker = False
    diag_checker = False
    cli_checker = False
    dic["disease"] = disease
    dic["location"] = fl
    dic["book_name"] = book_name
    return(dic,checker,diag_checker,cli_checker)

def reg_encoding(substr,j):
    # concode a chinese substring and try to find the matches within a string
    matched = substr      
    encoded = str(matched.encode("unicode_escape")).split("\\")
    encoded = [i for i in encoded if len(i)>=5]
    encoded = (".*\\"+"\\".join(encoded[:2])+".*\\"+encoded[2]).replace("'","")
    return(re.match(encoded,j))

def reg_encoding_front(substr,j):
    # concode a chinese substring and try to find the matches within a string
    matched = substr      
    encoded = str(matched.encode("unicode_escape")).split("\\")
    encoded = [i for i in encoded if len(i)>=5]
    encoded = ("\\"+encoded[0]+".{0,3}\\"+"\\".join(encoded[1:])+".{0,2}").replace("'","")
    return(re.match(re.compile(encoded),j))
def dic_comparison(dic1,dic2):
    #comparing two dictionaries, if all contents of the former one is included in the latter one, we update the first
    # dictionary, otherwise no change is made. This function is applied for iteratively updating the last element in the 
    # disease content dictionary collections, where we are not sure about where the lines of content would stop for the 
    # current disease. If a new disease dictionary is generated, the last dictionary will stop being updated and the 
    # new one would be appened and updated in further iterations.
    if dic1["disease"] != dic2["disease"]:
        return False
    for k in ["diagnosis","clinical","treatment"]:
        if k in dic1.keys() and k in dic2.keys():
            if dic1[k] in dic2[k]:
                dic1[k] = dic2[k]
            else:
                return False
        if k not in dic1.keys() and k in dic2.keys():
            dic1[k] = dic2[k]

diseases = [] ##
def get_results(contents,book_name,finallocation):
    """The main function for information extraction.
    Input: 
    1. contents: the list of list of lines in a book extracted from ocr. each element of contents 
is consist of a list of lines, representing a cut section/chapter from the original book 
    2. book_name: a string
    3. finallocation: a list of strings describing the exact chapters where the 
corresponding line in contentsoccurs in the book
    """ 
    #
    final = []
    dic = {}
    for content_id,(c,fl) in enumerate(zip(contents,finallocation)):
     #Iterating through all the sections from contens.
        identifier=[]
        if len(c) <= 20: continue
    # short sections are useless. Also we initialize the variables. The checker variables served as 
    #identifier for whether to continue appending lines for a certain content block or not
        checker = False
        diag_checker = False
        cli_checker = False
        treatment = ""
        diagnosis = ""
        clinical = ""
        disease = ""
        indicator = 0
#         dic={}
        diseaser_name_helper = 0
        for i,j in enumerate(c):
## for each line from a section of a book, we first match the line to a few regex patterns capturing the majority of
# the scenarios where a line serves as the sub-title disease name for the following lines.
            mres = re.match(".{0,2}\u4e00、|.{0,2}\u4e8c、|.{0,2}\u4e09、|.{0,2}\u56db、|.{0,2}\u4e94、|.{0,2}\u516d、\
            |.{0,2}\u4e03、|.{0,2}\u516b、|.{0,2}\u4e5d、|.{0,2}\u5341、",j)
            nres = re.match(".{0,2}\u4e00\)|.{0,2}\u4e8c\)|.{0,2}\u4e09\)|.{0,2}\u56db\)|.{0,2}\u4e94\)\
            |.{0,2}\u516d\)|.{0,2}\u4e03\)|.{0,2}\u516b\)|.{0,2}\u4e5d\)|.{0,2}\u5341\)",j)
            sres = re.match("^、.*\\n",j)
            pattern = re.compile('^\u7b2c.*\u90e8\u5206|^\u7b2c.*\u7ae0.*|^\u7b2c.*\u8282|^\u7b2c.*\u7bc7')
            result = re.match(pattern,j)
            j = j.strip()
            if i == 0:
                title = get_disease2(j)
                if title is not None:          
                    diseases.append(title)
                    disease = title
                    dic,checker,diag_checker,cli_checker = change(disease,fl,book_name)
                    treatment = ""
                    diagnosis = ""
                    clinical = ""                    
            elif indicator >= 0:
                if result is not None:
                    if len(j) <= 15 and "分册" in j:
                        continue
                    if len(final) > 0:
                        if final[-1]["disease"] == dic["disease"]:
                            dic_comparison(final[-1],dic)
                    if "篇" in j:
                        disease = j.split("篇",1)[1]
                    elif "章" in j:
                        disease = j.split("章",1)[1]
                    elif "节" in j:
                        disease = j.split("节",1)[1]
                    if disease != "":
                        dic,checker,diag_checker,cli_checker=change(disease,fl,book_name)
                        treatment = ""
                        diagnosis = ""
                        clinical = "" 

                elif not any(x in j for x in falsealarm):
     ## other scenarios, the matched pattern might be mistaken for a disease section title.
    ## to minimize the possibility of this false positive scenario, we want to gurantee that a "】" symbol would
    ## occur within the next 4 lines, otherwise it might be false alarm. Here we dont initialize the new dic yet 
    # upon detecting a match, instead we use a variable diseaser_name_helper to assist us for the storing and updating
    ### the number of lines that has through since we found a new matched pattern
                    if nres is not None:
                        disease = j.split(")",1)[1]
                    elif mres is not None or sres is not None:
                        disease = j.split("、",1)[1]
                    elif len(j) <= 10 and len(j) >= 2:
                        rresult = get_disease2(j)
                        if rresult == j.strip():
                            disease = rresult        
                    diseaser_name_helper += 1                
                                    
##如果通过直接整行匹配疾病得到的疾病 在接下来的一两行内没有出现"】"的符号，那么舍弃这个字典。
            if diseaser_name_helper >= 1 and "】" not in j:
                diseaser_name_helper += 1
            elif diseaser_name_helper >= 1 and "】" in j:
                diseaser_name_helper = 0
                dic = {}
                indicator = 0
                checker = False
                diag_checker = False
                cli_checker = False
                dic["disease"] = disease
                dic["location"] = fl
                dic["book_name"] = book_name
                treatment = ""
                diagnosis = ""
                clinical = ""
                   
            if diseaser_name_helper > 4:
                diseaser_name_helper = 0

                ## find the subsection patterns, start adding new lines to it and stop once a new "】" section is found.
                
            if (reg_encoding("治疗】",j) is not None) or (reg_encoding("治疔】",j) is not None) or reg_encoding_front("【治疗",j) is not None:
                checker = True
                indicator += 1
                treatment = ""
            elif ("】" in j or "【" in j) and "治疗" not in j and "治疔" not in j:
                if checker == True: 
                    dic["treatment"] = treatment
                checker = False
                
            if checker == True:
                treatment += j.strip()
                 
            if reg_encoding("诊断】",j) is not None or reg_encoding_front("【诊断",j) is not None:
                diag_checker = True
                indicator += 1
                diagnosis = ""
            elif ("】" in j or "【" in j) and "诊断" not in j:
                if diag_checker == True:
                    dic["diagnosis"] = diagnosis
                diag_checker = False
                
                
            if diag_checker ==True:
                diagnosis += j.strip()
                
                
            if reg_encoding("临床】",j) is not None or reg_encoding_front("【临床",j) is not None:
                clinical = ""
                indicator += 1
                cli_checker = True
            elif ("】" in j or "【" in j) and "临床" not in j:
                if cli_checker == True:
                    dic["clinical"] = clinical
                cli_checker = False
                
                
            if cli_checker == True:
                clinical += j.strip()
                    
            if indicator>=1:
        ##will append or update (depending on if the disease name is in the last element of the collection) 
        ##the dic to collections if it has found at least one content blocks.
                if len(diagnosis) > 1:
                    dic["diagnosis"] = diagnosis
                if len(treatment) > 1:
                    dic["treatment"] = treatment
                if len(clinical) > 1:
                    dic["clinical"] = clinical
#                 print("dic",dic)
                if len(final) == 0:
                    final.append(dic)
                elif final[-1]["disease"] == dic["disease"]:
                    dic_comparison(final[-1],dic)
                else:
                    final.append(dic)
    return(final)

class formater():
    def __init__(self,surtes):
        self.falsealarm = ["检查","表现","治疗","并发症","疼痛","死亡","检査","手术","临床","分类","。",";"]
        self.falsealarm_manual = ["表现","并发症","死亡","临床","分类","。",";"]
        with open("dictionaries/operation_single_suffix.json") as wf:
            single_suffix = json.load(wf)
        self.single_suffix = list(single_suffix.keys())
        self.suffix = ["测定","试验","实验","检查", "术", "造影", "放射治疗", "法", "监测"]
        self.pattern = re.compile('^\u7b2c.{1,3}\u90e8\u5206|^\u7b2c.{1,3}\u7ae0.*|^\u7b2c.{1,3}\u8282|^\u7b2c.{1,3}\u7bc7.*')
        self.mres = ".*\u4e00、|.*\u4e8c、|.*\u4e09、|.*\u56db、|.*\u4e94、|.*\u516d、|.*\u4e03、|.*\u516b、|.*\u4e5d、|.*\u5341、"
        self.nres = ".*\u4e00\)|.*\u4e8c\)|.*\u4e09\)|.*\u56db\)|.*\u4e94\)|.*\u516d\)|.*\u4e03\)|.*\u516b\)|.*\u4e5d\)|.*\u5341\)"
    def get_cutting(self,cutting_line,tlen):
        cutters = []
        for i,l in enumerate(cutting_line):
            if i == len(cutting_line)-1:
                cutters.append((l,tlen))
            else:
                cutters.append((l,cutting_line[i+1]))
        return cutters

    """return the matched disease name given a string. Different types of regex matching are combined,
    including matching by keyword for disease title identification; identifying the "、XXX疾病" pattern;
    identifying "三) XXX" or "三、XXX" pattern. Finally if no regex patterns is found, we search for 
    the entire disease list for maching
    """

    def get_treatment2(self,j):
        disease = None
        result = re.match(self.pattern,j)

        if result is not None:
            if "篇" in j:
                disease = j.split("篇",1)[1]
            elif '章'in j:
                disease = j.split("章",1)[1]
            elif "节" in j:
                disease = j.split("节",1)[1]
            elif "部分" in j:
                disease = j.split("部分",1)[1]
        elif re.match(self.mres,j) is not None:
            disease = j.split("、")[1]
        elif re.match(self.nres,j)  is not None:
            disease = j.split(")")[1]
        else:
            for it in surtes:
                if it in j:
                    disease = it
                    break
        if disease in exclusions:
            disease = None
        return disease


    def dic_init(self,disease,fl,book_name):
        ## Initializing disease content dictionary when a new disease name line is found 
        self.dic = {}
        self.indicator = 0
        self.checker = False
        self.diag_checker = False
        self.cli_checke r= False
        self.dic["measure"] = disease
        self.dic["location"] = fl
        self.dic["book_name"] = book_name


    def reg_encoding_twosided(self,substr,j):
        matchedfront = "【"+substr   
        matchedend = substr+"】"
        encoded_front = str(matchedfront.encode("unicode_escape")).split("\\")
        encoded_front = [i for i in encoded_front if len(i) >= 5]
        encoded_front = ("\\"+encoded_front[0] + ".{0,3}\\"+"\\".join(encoded_front[1:]) + ".{0,2}").replace("'","")
        encoded_end = str(matchedend.encode("unicode_escape")).split("\\")
        encoded_end = [i for i in encoded_end if len(i) >= 5]
        encoded_end = (".*\\"+"\\".join(encoded_end[:2]) + ".*\\" + encoded_end[2]).replace("'","")
        m1 = (re.match(re.compile(encoded_front),j))
        m2 = (re.match(re.compile(encoded_end),j))
        if m1 != None or m2 != None:
            return True
        else:
            return False
        
    def dic_comparison(self,dic1, dic2):
        if dic1["measure"] != dic2["measure"]:
            return False
        
        for k in ["indications","contradictions","procedures","complications"]:
            if k in dic1.keys() and k in dic2.keys():
                if dic1[k] in dic2[k]:
                    dic1[k] = dic2[k]
                else:
                    return False
            if k not in dic1.keys() and k in dic2.keys():
                dic1[k] = dic2[k]        

            
    def content_accumulator(self,keyname,j,keyword,contentstring,checkertype):
        if self.reg_encoding_twosided(keyword,j):
            self.checker = True
            self.indicator += 1
            contentstring = ""
            ## if checker is true: keeps accumulating the string until another 【 is encountered, 
            # when we stop accumulating and update the dictionary
        elif ("】" in j or "【" in j) and keyword not in j:
            if checkertype == True:
                self.dic[keyname]=contentstring
            checkertype = False
        if checkertype ==True:
            contentstring += j.strip()
#             print(contentstring)
        return(contentstring,checkertype)
    def append_final(self,final,dic_key):
        if self.indicator >= 1:
#             print(self.dic)
#             print(self.indications)
#             print("")
        ##will append or update (depending on if the disease name is in the last element of the collection) 
        ##the dic to collections if it has found at least one content blocks.
            if len(self.indications) > 1 and self.checker == True:
                self.dic[dic_key] = self.indications
            if len(final) == 0:
                final.append(self.dic)
            elif final[-1]["measure"] == self.dic["measure"]:
                self.dic_comparison(final[-1],self.dic)
            else:
                final.append(self.dic)
        return(final)
    def get_results(self,contents,book_name,finallocation,struc_keyword,dic_key):
        """The main function for information extraction.
        Input: 
        1. contents: the list of list of lines in a book extracted from ocr. each element of contents 
    is consist of a list of lines, representing a cut section/chapter from the original book 
        2. book_name: a string
        3. finallocation: a list of strings describing the exact chapters where the 
    corresponding line in contentsoccurs in the book
        """ 
        #测定 试验 实验 检查 术 造影 放射治疗 法 监测
        final=[]
        self.dic={}
        for content_id,(c,fl) in enumerate(zip(contents,finallocation)):
        #Iterating through all the sections from contens.
            identifier = []
            if len(c) <= 9: continue
        # short sections are useless. Also we initialize the variables. The checker variables served as 
        #identifier for whether to continue appending lines for a certain content block or not
            self.checker = False #适应证 适应证及临床意义 检查内容 检查内容及适应证      
            self.indicator = 0
            self.indications = ""
            self.complications = ""
            self.equipment = ""
            diseaser_name_helper = 0
            ## for each sentence in section
            for i,j in enumerate(c):
#                 if "第七节双乳头瓣移位术" in j:
#                     print("init",j)
                #
    #             if "节" in j:
    #                 print(j)
    ## for each line in a section from the book, we first match the line to a few regex patterns capturing the majority of
    # the scenarios where a line serves as the sub-title disease name for the following lines.

                sres = re.match("^、.*\\n",j)
                result = re.match(self.pattern,j)
                j = j.strip()
                if i == 0:
                    title = self.get_treatment2(j)


                    if title is not None:          
                        disease = title
                        if disease.endswith("的") and len(c[i+1]) <= 13:
                            disease += c[i+1].strip()
                        self.dic_init(disease,fl,book_name)
                        diseaser_name_helper += 1       
                elif self.indicator >= 0 and not any(x in j for x in self.falsealarm_manual):
                    if re.match(self.nres,j) is not None:
                        disease = j.split(")",1)[1]             
                    elif re.match(self.mres,j) is not None or sres is not None:
                        disease = j.split("、",1)[1]
                    elif len(j) <= 14 and len(j) >= 3:
                        rresult = self.get_treatment2(j)
                        if rresult == j.strip():
                            disease = rresult  
                            if disease.endswith(tuple(self.single_suffix)):
        #################append current dictionaries
                                if self.indicator >= 1:
                                    self.dic[dic_key] = self.indications
                                    final.append(self.dic)
                                self.dic_init(disease, fl, book_name)
                                diseaser_name_helper += 1



                            
                                              
    ##如果通过直接整行匹配疾病得到的疾病 在接下来的一两行内没有出现"】"的符号，那么舍弃这个字典。
#                 self.indications,self.checker=self.content_accumulator\
#                     (dic_key,j,struc_keyword,self.indications,self.checker)
            
                if self.reg_encoding_twosided(struc_keyword,j):
#                     print(disease,j)
                    self.checker = True
                    self.indicator += 1
                    self.indications = ""
#                     if "第七节双乳头瓣移位术" in j:
#                         print(self.checker)
#                             print("hey",[item for item in sect_id if item in line][0],disease)
                    ## if checker is true: keeps accumulating the string until another 【 is encountered, 
                    # when we stop accumulating and update the dictionary
#                     print(disease,self.checker)
                elif (("】" in j or "【" in j) and struc_keyword not in j) or (i==len(c)-1 and len(self.indications)>=1):
#                     print(disease,j)
                    if disease != self.dic["measure"]:
                        self.dic["measure"] = disease
                        self.dic["location"] = fl
                        self.dic["book_name"] = book_name
                    final = self.append_final(final,dic_key)
                    self.checker = False
                    diseaser_name_helper = 0
                    self.dic = {}
                    self.indicator = 0
                    self.checker = False
                    self.dic["measure"] = disease
                    self.dic["location"] = fl
                    self.dic["book_name"] = book_name
#                     if i==len(c)-1 and len(self.indications)>=1:
#                         self.dic[dic_key]=self.indications
#                         final.append(self.dic)
                    
                if self.checker == True:
                    self.indications += j.strip()
#                     print(self.indications)
                    
                if diseaser_name_helper > 8:
                    diseaser_name_helper = 0
                elif diseaser_name_helper >= 1 and "】" not in j:
                    diseaser_name_helper += 1
#                 elif diseaser_name_helper>=1 and "】" in j:  # if any section identifier is found, initialize new dic
#                     diseaser_name_helper=0
#                     self.dic={}
#                     self.indicator=0
#                     self.checker=False
#                     self.dic["measure"]=disease
#                     self.dic["location"]=fl
#                     self.dic["book_name"]=book_name
#                 print(i,self.checker)
                # self.complications,self.comp_checker=self.content_accumulator\
                #     ("complications",j,"并发症",self.complications,self.comp_checker)      

        return(final)
def generate_dic(direc = "books_m/", bookname="all"):
    ## 针对操作手册
    res_operation = []
    form = formater(surtes)
    counter = 0

    for book in os.listdir(direc):
        if bookname == "all":
            pass
        else:
            if book != bookname:
                continue
        counter += 1
        if ".ipyn" in book:
            continue
        cutting_line = []
        chapters = []
        chapter_num = []
        section_num = []
        chapter_string = []
    #     chapter_linenbr = []
        with open(direc + book,encoding = "utf-8") as textfile:
            lines = textfile.readlines()
        contents_starter = 0
        pattern = re.compile('^\u7b2c.{1,3}\u90e8\u5206|^\u7b2c.{1,3}\u7ae0.*|^\u7b2c.{1,3}\u8282|^\u7b2c.{1,3}\u7bc7.*')
        part = ""
        for i,line in enumerate(lines):

            if "目录" in line:
                contents_starter = i+1
            result = re.match(pattern,line)
            if result is not None:
                if [item for item in sect_id if item in line] != []:
                    item = [item for item in sect_id if item in line][0]
#                         chapter_string.append(line)
                    if item == "部分" or item == "篇":
                        if line in chapter_string:
                            continue
                        else:
                            cutting_line.append(i)
                            disease = line.split(item,1)[1]
                            chapters.append(disease)
                            chapter_string.append(line)
                            part = line.strip()
                            chapter = ""
                            section = ""

                    elif item == "章":
                        if line in chapter_string:
                            continue
                        else:
                            cutting_line.append(i)
                            disease = line.split(item,1)[1]
                            chapters.append(disease)
                            chapter_string.append(line)
                            chapter = line.strip()
                            section = ""
                    elif item == "节":

                        cutting_line.append(i)
                        disease = line.split(item,1)[1]
#                         if "第七节双乳头瓣移位术" in line:
#                             print("hey",[item for item in sect_id if item in line][0],disease)
                        chapters.append(disease)
#                         print(line)
                        section = line.strip()
                    joined = part + "|" + chapter + "|" + section
                    chapter_num.append(joined)
#         print(chapters)
        cutters = form.get_cutting(cutting_line,len(lines))
        contents = []
        for i,j in zip(cutters,chapters):
            contents.append(lines[i[0]:i[1]])

        paragraphs = []
        for c in contents:
            content = ""
            for j in c:
                content += j.strip()
            paragraphs.append(content)

    #     finallocation=[(cn.strip()+"|"+sn.strip()) for cn,sn in zip(chapter_num,section_num)]
    #     print(finallocation)
        new_items = form.get_results(contents,book,chapter_num,"适应","indications")
#         print(new_items)
        new_items += form.get_results(contents,book,chapter_num,"目的","indications")
#         print(len(new_items))
        new_items += form.get_results(contents,book,chapter_num,"临床","indications")
#         print(len(new_items))
        new_items += form.get_results(contents,book,chapter_num,"环境及器械要求","equipments")
#         print(len(new_items))
        new_items += form.get_results(contents,book,chapter_num,"操作","procedures")
#         print(len(new_items))
        new_items += form.get_results(contents,book,chapter_num,"方法","procedures")
        new_items += form.get_results(contents,book,chapter_num,"准备","preparation")
        new_items += form.get_results(contents,book,chapter_num,"术前","preparation")
        new_items += form.get_results(contents,book,chapter_num,"并发症","complications")
        new_items += form.get_results(contents,book,chapter_num,"不良反应","adverse")
        new_items += form.get_results(contents,book,chapter_num,"禁忌","contraindiction") 
        new_items += form.get_results(contents,book,chapter_num,"麻醉","anaesthesia") 
        new_items += form.get_results(contents,book,chapter_num,"术后","afterwards")
        new_items += form.get_results(contents,book,chapter_num,"设备","equipments")
        new_items += form.get_results(contents,book,chapter_num,"检查","exams")
        new_items += form.get_results(contents,book,chapter_num,"检查内容","procedures")
        print("{}: {} new items".format(book,len(new_items)))
        res_operation += new_items
    return(res_operation)
    
if __name__ == "__main__":
    res2=[]
    counter = 0
    sect_id = ["篇","部分","章","节"] ## key characters serve as section identifier

    """操作手册"""
    pre = utillist.fetch_dic(stype="PRE")
    tes = utillist.fetch_dic(stype="TES")
    sur = utillist.fetch_dic(stype="SUR")
    surtes = sur+tes+pre
    surtes.sort(key=len)
    surtes = surtes[::-1]
    indicator_sets = defaultdict(int)
    for book in os.listdir(manuals):
        if ".ipyn" in book:
            continue
        with open(manuals+book,encoding="utf-8") as textfile:
            lines = textfile.readlines()
        for i,line in enumerate(lines):
            found_ind = re.findall('\【(.*?)\】',line)
            if len(found_ind) >= 1:
    #             if "治疗" not in found_ind[0]\
    #             and "诊断" not in found_ind[0] and "临床" not in found_ind[0]:
                indicator_sets[found_ind[0]] += 1


    results = generate_dic(bookname="all")
    res_operation_df = pd.DataFrame(results)
    
    brochure_dfgp = res_operation_df.groupby(["measure","location","book_name"])
    uniqueindexes = []
    for i,j,k in zip(res_operation_df.measure,res_operation_df.location,res_operation_df.book_name):
        if (i,j,k) not in uniqueindexes:
            uniqueindexes.append((i,j,k))
    finalresult = []
    cols = res_operation_df.columns
    alldics = []
    for ind in uniqueindexes:
        dicappended = {}
        newdf = brochure_dfgp.get_group(ind)
        dicappended[cols[0]] = ind[0]
        dicappended[cols[1]] = ind[1]
        dicappended[cols[2]] = ind[2]
        for colname in cols[3:]:
            for d in newdf[colname]:
                if type(d) is str:
                    dicappended[colname]=d
        alldics.append(dicappended)
    operation_df = pd.DataFrame(alldics)
    operation_df = operation_df.fillna("")
    # operation_df_2 = operation_df[operation_df.indications.str.contains("【临床表现")]
    # c = operation_df[~operation_df.indications.str.contains("【临床表现")]
    operation_df.to_csv("preprocessed/manual_sub.csv",encoding="utf-8-sig")

    # titles = list(res_operation_df[res_operation_df.book_name=="激光医学分册.txt"].measure)
    # indications = list(res_operation_df[res_operation_df.book_name=="激光医学分册.txt"].indications)
    # results = []
    # for x,y in zip(titles,indications):
    #     results.append(y+"。"+x)
    # res_operation_df.loc[res_operation_df.book_name=="激光医学分册.txt","indications"] = results
    res_operation_df.to_csv(preprocessed_manual,encoding="utf-8-sig")


    """临床指南"""

    for book in os.listdir(clinicals):
        counter += 1
        if ".ipyn" in book:
            continue
        cutting_line = []
        chapters = []
        chapter_num = []
        section_num = []
        chapter_string = []
    #     chapter_linenbr=[]
        with open(clinicals+book,encoding="utf-8") as textfile:
            lines = textfile.readlines()
        contents_starter = 0
        pattern = re.compile('^\u7b2c.{1,3}\u90e8\u5206|^\u7b2c.{1,3}\u7ae0.*|^\u7b2c.{1,3}\u8282|^\u7b2c.{1,3}\u7bc7.*')
    #     print()
    #     print(book)
        part = ""
        for i,line in enumerate(lines):
            if "目录" in line:
                contents_starter = i+1
            result = re.match(pattern,line)
            if result is not None:
                for item in sect_id:
                    if item in line:
                        chapter_string.append(line)
                        cutting_line.append(i)
                        disease = line.split(item,1)[1]
                        chapters.append(disease)
                        if item == "部分" or item == "篇":
                            part = line.strip()
                            chapter = ""
                            section = ""
                        elif item == "章":
                            chapter = line.strip()
                            section = ""
                        elif item == "节":
                            section = line.strip()
                        joined = part + "|" + chapter + "|" + section
                        chapter_num.append(joined)
        cutters = get_cutting(cutting_line,len(lines))
        contents = []
        for i,j in zip(cutters,chapters):
            contents.append(lines[i[0]:i[1]])
        paragraphs = []
        for c in contents:
            content = ""
            for j in c:
                content += j.strip()
            paragraphs.append(content)
    #     finallocation=[(cn.strip()+"|"+sn.strip()) for cn,sn in zip(chapter_num,section_num)]
    #     print(finallocation)
        new_items = get_results(contents,book,chapter_num)

        print("{}: {} new items".format(book,len(new_items)))
        res2 += new_items
    res2 = pd.DataFrame(res2)
    # res2.to_csv("preprocessed/clinical_col_1.csv")
#     res2.columns = list(res2.columns)[:3]+["overview","subtype1","subtype2"]
#     res2.to_csv("preprocessed/clinical_col_2.csv")
    treats = pd.DataFrame(res2).drop_duplicates()
    treats2 = treats.dropna(subset = ["clinical","diagnosis","treatment"],how="all")
    treats2 = treats2[~(treats2.disease.str.contains("治疗")|treats2.disease.str.contains("手术")|treats2.disease.str.contains("。")|\
                  treats2.disease.str.contains("诊断")|treats2.disease.str.contains("临床")|treats2.disease.str.contains("章"))]
    treats2 = treats2[~(treats2.disease.str.contains(":")|treats2.disease.str.contains("、")|treats2.disease.str.endswith("。")|\
                  treats2.disease.str.contains(",")|treats2.disease.str.contains("分类")|treats2.disease.str.endswith("期"))]
    
    treats2.disease.astype(str)
    dl = list(set(treats2.disease)).sort(key=len)
    dl2 = list(set(treats2.disease)).sort(key=len)
    treats2 = treats2.dropna(subset=["disease"])
    newdiseasename = []
    for j in treats2.disease:
        if "(" in j:
            nd = re.sub("[\(\[].*?[\)\]]", "", j)
            if ("(") in nd:
                nd = nd.split("(")[0]
    #             print(type(nd))
    #         print(nd,"|",j)
            
            newdiseasename.append(nd)
        else:
            newdiseasename.append(j.split())
    treats2.disease = newdiseasename
    newdiseasename = []
    for j in treats2.disease:
        if type(j) == list:
            print(j)
            if len(j) == 0:
                newdiseasename.append("")
            else:
                toa = j[0]
                newdiseasename.append(toa)
            
        else:
            newdiseasename.append(j)
    treats2.disease=newdiseasename
    disexclu=[]
    for j in treats2.disease:
        if len(j)<3:
            if j not in dis:
                disexclu.append(j)
    treats = treats2[~treats2.disease.isin(disexclu)]
    treats.to_csv(preprocessed_clinical)

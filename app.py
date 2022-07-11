
from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle as p
import json


app = Flask(__name__)

import numpy as np
import os
import re
import json
import jieba
import jieba.posseg
from multi_source_ner import ChineseNER

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
    #preds = model.predict_oneline(sentence2)
    print("something`")
    data = request.get_json(force=True)
    print("data", data)
    sentence = data['sentence']
    #s1=data[1]
    #return(str(data))
    prediction = model.predict_oneline(sentence)
    return(jsonify(prediction))
    #returnonifyjsonify(prediction)

if __name__ == '__main__':
    modelfile = 'models/params_o_cnn_a.pkl'
    model = ChineseNER("predictonly", modelfile)
    app.run(host='192.168.4.34', port = 4100 ,debug=True)

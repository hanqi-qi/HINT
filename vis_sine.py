import numpy as np
import re
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
latex_special_token = ["!@#$%^&*()"]

def label2symbol(label):
    if label >0:
        symbol = "$+$1"
    else:
        symbol = "$-$1"
    return symbol

def generate(doc_sen, doc_word_att, doc_sen_att,sen_labels,latex_file, sen_color='green',word_color='red',pre_label=None,target=None):
    stopwords = set(STOPWORDS)
    stopwords.update("i, my, me, he, his,him, she, her, they")
    assert(len(doc_sen) == len(doc_sen_att))
    doc_id = re.findall(r'\d+',latex_file)
    # if rescale_value:
    #     attention_list = rescale(attention_list)
    #     word_num = len(text_list)
    #     text_list = clean_word(text_list)
    with open(latex_file,'w') as f:
        f.write(r'''\documentclass{article}
\usepackage{tikz,lipsum,lmodern}
\usepackage[most]{tcolorbox}
\usepackage{caption}
\usepackage{subcaption}
\begin{document}
\begin{tcolorbox}[colback=yellow!5!white,colframe=yellow!50!black,colbacktitle=yellow!75!black,''')
        f.write('title=Document '+(doc_id[0]))
        f.write(r''',fonttitle=\bfseries]'''+'\n')
        string = ""
        senatt_arr = rescale(doc_sen_att)
        for sen_id in range(len(doc_sen)):
            string += r"\colorbox{%s!%s}{S"%(sen_color,senatt_arr[sen_id])+str(sen_id+1)+"} "
            att_arr = rescale(doc_word_att[sen_id])
            for word_id in range(len(doc_sen[sen_id])):
                if doc_sen[sen_id][word_id] in latex_special_token[0]:
                    word = "\\"+doc_sen[sen_id][word_id].strip()
                else:
                    word = doc_sen[sen_id][word_id].strip()
                if att_arr[word_id]>20 and doc_sen[sen_id][word_id] not in stopwords:
                    string += r"\colorbox{%s!%s}{"%(word_color, att_arr[word_id])+ word+"} "
                else:
                    string += word+ " "
            string += r'(\textsf{'+str(label2symbol(sen_labels[sen_id]))+"})" +r"\\"+"\n"
        string += r"\tcblower"+"\n"
        string += r"Predict Label: \textsf{"+str(label2symbol(pre_label))+r"} \\"+"\n"
        string += r"GroundTruth Label: \textsf{"+str(label2symbol(target))+"}"+"\n"
        string += r"\end{tcolorbox}"+"\n"
        f.write(string+'\n')
        # f.write(r"\begin{figure}[h]"+"\n"+r"\centering"+"\n"+
        # r"\begin{subfigure}[b]{0.48\textwidth}"+"\n"+
        # r"\centering"+"\n"+
        # r"\includegraphics[width=\textwidth,trim={1cm 0cm 1cm 1cm},clip]")
        # pic_str = "{"+"./doc_wc/IMDB_DocID{}_wordcloud_casestudy.png".format(doc_id[0])+"}"
        # f.write(pic_str)
        # f.write("\n"+r"\end{subfigure}"+"\n"+
        #         r"\hfill"+"\n"+
        #         r"\begin{subfigure}[b]{0.48\textwidth}"+"\n"+
        #         r"\centering"+"\n"
        #         r"\includegraphics[width=\textwidth,trim={1cm 0cm 1cm 1cm},clip]")
        # pic_str = "{"+"./doc_wc/IMDB_DocID{}_t2_wordcloud.png".format(doc_id[0])+"}"
        # f.write(pic_str+"\n")
        # f.write(r"""\end{subfigure}"""+"\n"+r"""\end{figure}""")
        f.write("\n"+r'''\end{document}''')
        f.close()

def rescale(input_list):
	the_array = np.asarray(input_list)
	the_max = np.max(the_array)
	the_min = np.min(the_array)
	rescale = (the_array - the_min)/(the_max-the_min)*100
	return rescale.tolist()


def clean_word(word_list):
	new_word_list = []
	for word in word_list:
		for latex_sensitive in ["\\", "%", "&", "^", "#", "_",  "{", "}"]:
			if latex_sensitive in word:
				word = word.replace(latex_sensitive, '\\'+latex_sensitive)
		new_word_list.append(word)
	return new_word_list

def gen_hard_rats(doc,doc_word_att,level,doc_wpos_id):
    """write out the test_decode.json as original ERASER project:
    1) use level to decide how many percentage important words will be selected as hard rats.
    2) len(doc) == len(doc_att) == len(truth_rat) truth_rat is the same length as doc, element is 1/0 represents rats or not"""
    rat_list = []
    for sen_id in range(len(doc)):
        assert len(doc[sen_id]) == len(doc_word_att[sen_id]) == len(doc_wpos_id[sen_id])
        deleted_num = max(int(len(doc_word_att[sen_id])*level),1)
        important_idx = list(np.array(doc_word_att[sen_id]).argsort()[-deleted_num:][::-1])
        # threshold = np.percentile(np.array(doc_word_att[sen_id]),100*(1-level))
        for i in range(len(doc_word_att[sen_id])):
            if i in important_idx:
            # if doc_word_att[sen_id][i]>threshold:
                print(doc_wpos_id[sen_id][i],doc[sen_id][i])
                rat_list.append({'start_token': int(doc_wpos_id[sen_id][i]), 'end_token': int(doc_wpos_id[sen_id][i])+1})
    return rat_list
    
# if __name__ == '__main__':
#     ## This is a demo:
#     import random
#     random.seed(42)
#     doc = ["the USS Ronald Reagan", "an aircraft carrier docked in Japan", "during his tour of the region", "vowing to defeat any attack and meet any use of conventional or nuclear weapons with an overwhelming and effective American response", "North Korea and the US have ratcheted up tensions in recent weeks and the movement of the strike group had raised the question of a pre-emptive strike by the US.","On Wednesday, Mr Pence described the country as the most dangerous and urgent threat to peace and security","in the Asia-Pacific."]
#     doc_word_att = []
#     doc_sen_att = []
#     sen_labels = []
#     for sen in doc:
#         words = sen.split()
#         word_num = len(words)
#         sen_word_att = [(x+1.)/word_num*100 for x in range(word_num)]
#         sen_labels.append(word_num%2)
#         random.shuffle(sen_word_att)
#         doc_word_att.append(sen_word_att)
#     doc_sen_att = [(x+1.)/len(doc)*100 for x in range(len(doc))]
#     word_color = 'red'
#     sen_color = 'green'

#     generate(doc, doc_word_att, doc_sen_att,sen_labels,"vis_sine.tex", sen_color,word_color)
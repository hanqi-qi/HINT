from sklearn.metrics import cohen_kappa_score
y1_list = open("/home/hanq1yanwarwick/SINE/explain_metric/human_score_xw.txt").readlines()
y1 = []
for y in y1_list:
    y1.append(y)
y2_list = open("/home/hanq1yanwarwick/SINE/explain_metric/human_score_zy.txt").readlines()
y2 = []
for y in y2_list:
    y2.append(y)
#hq_xw: HAN_IMDBcorrectness 0.234/hq_xw: HAN_IMDBfaithfulness: 0.173/hq_xw: HAN_IMDBinformative: 0.2143
print(cohen_kappa_score(y1, y2))
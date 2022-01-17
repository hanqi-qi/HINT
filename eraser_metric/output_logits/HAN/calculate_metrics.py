import pandas as pd
import numpy as np
import math 
metric = "sset"
result =  {}
aopc_result= 0
for bin in [0,1,2,3,4]:
    f0 = pd.read_csv("/home/hanq1yanwarwick/SINE/eraser_metric/output_logits/HAN/prediction_results_original_999bin_selfexplain.csv",index_col="idx")
    f1 = pd.read_csv("/home/hanq1yanwarwick/SINE/eraser_metric/output_logits/HAN/prediction_results_{}_{}bin_selfexplain.csv".format(metric,bin),index_col="idx")
    combine_df = pd.concat([f0, f1], axis=1)
    #use the logit in correct prediction label
    diff = []
    correct2error=0
    for i in range(len(combine_df)):
        # if combine_df.iloc[i]["base_pre"] != combine_df.iloc[i]["base_label"]:
        if True:
            # if combine_df.iloc[i]["pre"] != combine_df.iloc[i]["base_label"]:
                # correct2error +=1
            if float(combine_df.iloc[i]["base_pre"]) == 1.0:
                # base_logits = math.exp(combine_df.iloc[i]["base_pos_logits"])/(math.exp(combine_df.iloc[i]["base_pos_logits"])+math.exp(combine_df.iloc[i]["base_neg_logits"]))
                # logits = math.exp(combine_df.iloc[i]["pos_logits"])/(math.exp(combine_df.iloc[i]["pos_logits"])+math.exp(combine_df.iloc[i]["neg_logits"]))
                # diff.append(base_logits-logits)
                diff.append(combine_df.iloc[i]["base_pos_logits"]-combine_df.iloc[i]["pos_logits"])
            elif float(combine_df.iloc[i]["base_pre"]) == 0:
                # base_logits = math.exp(combine_df.iloc[i]["base_neg_logits"])/(math.exp(combine_df.iloc[i]["base_pos_logits"])+math.exp(combine_df.iloc[i]["base_neg_logits"]))
                # logits = math.exp(combine_df.iloc[i]["neg_logits"])/(math.exp(combine_df.iloc[i]["pos_logits"])+math.exp(combine_df.iloc[i]["neg_logits"]))
                # diff.append(base_logits-logits)
                diff.append(combine_df.iloc[i]["base_neg_logits"]-combine_df.iloc[i]["neg_logits"])
    # if metric == "cset":
    # compeleteness = (combine_df["base_pos_logits"]-combine_df["pos_logits"]).values
    bin_result = sum(diff)/len(diff)
    print(len(diff))
    # print(len(diff))
    # print(correct2error)
    # if metric == "sset":
        # sufficiency = (combine_df["base_pos_logits"]-combine_df["pos_logits"]).values
    result["bin{}".format(bin)] = bin_result
    aopc_result += bin_result
print("The results of the %s"%metric)
print(result)
print(aopc_result/5)
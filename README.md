# HINT (SINE is older version)

### Two stage process for long document classification and interpretable evaluation <br>
 1) two paralle sentence representation learning modules: <br>
      a) att-lstm b) tfidf prior vae <br>
 2) full-connected GAT <br>
 
 ##### Baselines: <br>
 [VMASK](https://arxiv.org/abs/2010.00667)<br> [HAN](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf)
 
 #### Datasets
 IMDB, YELP, Guardian News
 
 #### Interpretable evaluation
 1) **soft metrics: Completeness and Sufficiency measuring the prediction changes after removing the important words.** <br>
    we create two folds for the text after removing the important words as cset, and another fold for extracted important words as sset. Calculation found in [here](./eraser_metrics/)
 3) **hard metrics/aggrement with human annotated rationales.** <br>
  Input 1: annotated text spans Input 2: identified important words by model. Details found in [Here](./explain_metric/metrics.py) These metrics include partial span-level match and token-level match.

#### Visualization (Highlight text by. attention weight)

We also conduct human evaluation for the model generated interpretations and they are displayed in seperate pdf (generated by latex) for each document. It contains 
explains at the **word level**, where the label-dependent words extracted by _Context Representation Learning_ are highlighted in yellow, while the topic-associated words identified by _Topic Representation Learning_ are highlighted in blue; alsp at the **sentence level** . So the python code for generate attention-highlight text latex can be as a good template. Code is [vis_sine.py](vis_sine.py)

![Example is Here](ex.png?raw=true "Title")

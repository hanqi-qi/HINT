from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from matplotlib import pyplot as plt
import random

topic_vec = [37, 37, 42, 44, 41, 10, 29, 41]
topic_weight = {37: 0.2533232271671295, 42: 0.12099681049585342, 44: 0.13011407852172852, 41: 0.25503627210855484, 10: 0.11790172755718231, 29: 0.12262789905071259}
wc_dict = {}
words = ['thanks', 'much', 'dr', 'reynolds', 'awesome','tough', 'one', 'dig', 'part', "n't", 'fall']
atts = [0.284031954, 0.3660977, 0.126630897, 0.14689734, 0.40583006,0.15560299, 0.058870535, 0.1087861866, 0.45824355, 0.07178907, 0.058870535, 0.10876147]
for att, word in zip(atts,words):
    wc_dict[word] = att

# wc_dict =  {"director":0.3,"actor":0.3,"dictator":0.2,"manipulative":0.28,"president":0.12,"Ringo":0.1,"politician":0.15,"kitsch":0.13,"nuclear":0.08}
# wc_dict =  {"trauma":0.3,"pain":0.26,"injurie":0.2,"insomnia":0.28,"depression":0.18,"stress":0.2,"alcoholism":0.25,"anxiety":0.12,"athletic":0.1,"hopeless":0.1}
# wc_dict =  {"inaccurate":0.3,"inconsistent":0.3,"incompetent":0.2,"demeanor":0.28,"stubborn":0.18,"ineffective":0.2,"pinned":0.25,"sloppy":0.1,"smirk":0.08}
# sent = "only thing enjoy Pink Floyd's wonderful soundtrack, too good for stereotypical waste like."
# wc_dict = {}
# for word in sent.split():
#     if word == "wonderful":
#         wc_dict[word] = 0.4
#     elif word == "soundrack":
#         wc_dict[word] = 0.3
#     elif word == "waste":
#         wc_dict[word] = 0.2
#     else:
#         wc_dict[word] = random.choice([0.03,0.02,0.01])

wordcloud = WordCloud(background_color="white")
wordcloud.generate_from_frequencies(frequencies=wc_dict)
plt.figure()
# plt.title("Topic from {} with weight {}".format("S4",round(1-0.5044-0.2435,4)))
plt.axis("off")
plt.imshow(wordcloud, interpolation="bilinear")
plt.savefig("./intro_pic/case3_t41.png",dpi=300)
plt.close()
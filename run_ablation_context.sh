for dataset in "yelp" "imdb" "guardian_news"
do
    for context_att in "0"
    do
        bash run_main2.sh $dataset "bayesians" $regularization $context_att "tfidf" >> log/${dataset}_ContextAtt${contextatt}.out 2>&1
    done
done
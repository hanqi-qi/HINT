for dataset in "yelp" "imdb" "guardian_news"
do
    for regularization in "0"
    do
        bash run_main2.sh $dataset "bayesians" $regularization 1 "tfidf" >> log/${dataset}_Reg${regularization}.out 2>&1
    done
done
for dataset in "yelp" "imdb" "guardian_news"
do
    for topic_learning in "autoencoder"
    do
        bash run_main2.sh $dataset $topic_learning 1 1 "tfidf" >> log/${dataset}_autoencoder.out 2>&1
    done
done
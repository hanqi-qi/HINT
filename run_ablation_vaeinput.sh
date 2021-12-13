for dataset in "yelp" "imdb" "guardian_news"
do
    for topic_weight in "average"
    do
        bash run_main2.sh $dataset "bayesians" 1 1 $topic_weight >> log/${dataset}_topicweight${topic_weight}.out 2>&1
    done
done
for dataset in "imdb" "guardian_news" "yelp" 
do
    for seed in "0"  "10"
    do
        bash run_main.sh $dataset $seed >>log/${dataset}_seed${seed}.out 2>&1
    done
done
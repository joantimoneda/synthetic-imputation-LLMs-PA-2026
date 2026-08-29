SemEval-2016 raw data (not redistributed)

The raw tweet files come from the SemEval-2016 Task 6 (Detecting Stance in
Tweets) distribution (Mohammad et al. 2016) and are not redistributed here.
"0. read_sentiment_data.py" expects the following four files in this folder,
all in the original tab-separated format (ID, Target, Tweet, Stance,
Opinion towards, Sentiment):

    trainingdata-all-annotations.txt          Task A training set (original name)
    testdata-taskB-all-annotations.txt        Task B test set (original name)
    sem_eval_2016_stance_sentiment_test.txt   Task A test set (renamed)
    sem_eval_2016_stance_sentiment_train.txt  Task A trial set (renamed)

The processed file the pipeline uses, all_sentiment_true.xlsx, is included in
the parent data folder, so the raw files are only needed to re-run "0." from
scratch.

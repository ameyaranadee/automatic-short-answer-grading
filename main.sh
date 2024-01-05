python -W ignore run.py \
--dataframe "./data/Data.csv" \
--qa_dataframe "./data/QA1.csv" \
--qd_train True \
--train_test_split 0.25 \
--w2v_baroni_dir "./data/EN-wform.w.5.cbow.neg10.400.subsmpl.txt" \
--qnum_colname "number" \
--que_colname "Questions" \
--ans_colname "Answers" \
--text_colname "Texts" \
--score_colname "Score" \
--regressor "RandomForestRegressor" \
${@:1}
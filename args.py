import argparse

def get_parser():

    parser = argparse.ArgumentParser()

    # parser.add_argument('--train', action='store_true')

    # Data directories
    parser.add_argument('--dataframe', type=str, required=True)
    parser.add_argument('--qa_dataframe', type=str, default=None)
    
    # Embeddings
    parser.add_argument('--w2v_baroni', default=True)
    parser.add_argument('--w2v_baroni_dir', type=str, default="./data/EN-wform.w.5.cbow.neg10.400.subsmpl.txt")

    # Column names
    parser.add_argument('--qnum_colname', type=str, required=True)
    parser.add_argument('--que_colname', type=str, required=True)
    parser.add_argument('--ans_colname', type=str, required=True)
    parser.add_argument('--text_colname', type=str, required=True)
    parser.add_argument('--score_colname', type=str, required=True)

    # Tasks
    parser.add_argument('--question_demoting', default=None)
    parser.add_argument('--qd_train', default=None)

    # Training
    parser.add_argument('--regressor', type=str, default='RandomForestRegressor')
    parser.add_argument('--train_test_split', type=float, default=0.25)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--max_iter', type=float, default=900)

    # Checkpoints
    parser.add_argument('--download_dir', type=str, default='./downloads/')

    return parser
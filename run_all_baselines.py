import os

cmds = [

    # 1. Transformer + Gaussian
    """python main.py eval-model 
        --jsonl output/jsonl_fbd/test.jsonl 
        --model transformer 
        --head gauss 
        --past_len 8 --future_len 60 
        --d_model 256 --nhead 8 --layers 4 
        --weights output/weights/trans_gauss.pt 
        --device cuda:0 
        --batch_size 256 
        --horizons 30 60 
        --out output/aaafinal_output/report_gauss.json""",

    # 2. Transformer + MDN Gaussian
    """python main.py eval-model 
        --jsonl output/jsonl_fbd/test.jsonl 
        --model transformer 
        --head mdn_gauss --mdn_k 5 
        --past_len 8 --future_len 60 
        --d_model 256 --nhead 8 --layers 4 
        --weights output/weights/trans_mdn_gauss.pt 
        --device cuda:0 
        --batch_size 256 
        --horizons 30 60 
        --out output/aaafinal_output/report_mdn_gauss.json""",

    # 3. Transformer + MDN StudentT
    """python main.py eval-model 
        --jsonl output/jsonl_fbd/test.jsonl 
        --model transformer 
        --head mdn_student_t --mdn_k 5 
        --past_len 8 --future_len 60 
        --d_model 256 --nhead 8 --layers 4 
        --weights output/weights/trans_mdn_student_t.pt 
        --device cuda:0 
        --batch_size 256 
        --horizons 30 60 
        --out output/aaafinal_output/report_mdn_student_t.json""",

    # 4. LSTM baseline
    """python main.py eval-model 
        --jsonl output/jsonl_fbd/test.jsonl 
        --model lstm 
        --head student_t 
        --past_len 8 --future_len 60 
        --d_model 256 
        --weights output/weights/lstm_student_t.pt 
        --device cuda:0 
        --batch_size 256 
        --horizons 30 60 
        --out output/aaafinal_output/report_lstm.json""",
]

os.makedirs("output/aaafinal_output", exist_ok=True)

for c in cmds:
    print("\n\n[RUN] ", c)
    os.system(c)

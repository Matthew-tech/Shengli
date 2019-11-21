import pickle as pkl

with open("result/cnn_target/pred_line_1899to1908.pkl", 'rb') as f1:
    d = pkl.load(f1)
    print(d[:10])
import argparse
from preprocessor import do_data
from trainer import do_train
from tester import do_test

parser = argparse.ArgumentParser()
parser.add_argument('--do_data', action='store_true')
parser.add_argument('--do_train', action='store_true')
parser.add_argument('--do_test', action='store_true')
parser.add_argument('--train_data')
parser.add_argument('--test_data')
parser.add_argument('--max_len', type=int, default=25)
parser.add_argument('--max_epoch', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--valid_split_rate', type=int, default=0.2)
parser.add_argument('--ncontrols', type=int, default=2)
parser.add_argument('--emb_dim', type=int, default=128)
parser.add_argument('--hid_dim', type=int, default=128)


args = parser.parse_args()


if args.do_data:
    do_data(args.train_data, args.test_data, args.valid_split_rate, args.max_len)

if args.do_train:
    do_train("../data/train_data.pkl", "../data/valid_data.pkl", "../data/dictionary.pkl",
             args.max_len, args.ncontrols, args.emb_dim, args.hid_dim, args.max_epoch, args.batch_size)

if args.do_test:
    do_test("../model1/model.pkl.9", "../data/test_data.pkl", "out.txt", "../data/dictionary.pkl", args.emb_dim, args.hid_dim)

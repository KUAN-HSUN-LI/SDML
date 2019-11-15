import torch
from model import get_model
from dataset import SentenceDataset
from predict import make_prediction
from utils import load_pkl


def do_test(model_ckpt_path, test_data_path, result_path, word_dict_path, emb_dim, hid_dim):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    word_dict = load_pkl(word_dict_path)
    PAD_IDX = word_dict.word2idx("<PAD>")
    print("load data...")
    testData = SentenceDataset(load_pkl(test_data_path), word_dict, PAD_IDX, training=False)
    print("load model...")
    model = get_model(word_dict.get_len(), word_dict.get_len(), emb_dim, hid_dim, device)
    model.load_state_dict(torch.load(model_ckpt_path))
    model.to(device)
    print("predicting...")
    make_prediction(model, testData, word_dict, result_path, device)

def remove_info(dataset):
    dataset.drop('Title', axis=1, inplace=True)
    dataset.drop('Categories', axis=1, inplace=True)
    dataset.drop('Created Date', axis=1, inplace=True)
    dataset.drop('Authors', axis=1, inplace=True)
    return dataset


def plot(save_dir):
    import matplotlib.pyplot as plt
    import json

    with open('../model/%s/history.json' % save_dir, 'r') as f:
        history = json.loads(f.read())

    train_loss = [l['loss'] for l in history['train']]
    valid_loss = [l['loss'] for l in history['valid']]
    train_f1 = [l['f1'] for l in history['train']]
    valid_f1 = [l['f1'] for l in history['valid']]

    plt.figure(figsize=(7, 5))
    plt.title('Loss')
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='valid')
    plt.legend()
    plt.savefig('../model/%s/loss.png' % save_dir)

    plt.figure(figsize=(7, 5))
    plt.title('F1 Score')
    plt.plot(train_f1, label='train')
    plt.plot(valid_f1, label='valid')
    plt.legend()
    plt.savefig('../model/%s/f1_score.png' % save_dir)

    print('Best F1 score ', max([[l['f1'], idx] for idx, l in enumerate(history['valid'])]))


def SubmitGenerator(prediction, sampleFile, public=True, filename='prediction.csv'):
    """
    Args:
        prediction (numpy array)
        sampleFile (str)
        public (boolean)
        filename (str)
    """
    import pandas as pd

    sample = pd.read_csv(sampleFile)
    submit = dict()
    submit['order_id'] = list(sample.order_id.values)
    redundant = len(sample) - prediction.shape[0]
    if public:
        submit['THEORETICAL'] = list(prediction[:, 0]) + [0]*redundant
        submit['ENGINEERING'] = list(prediction[:, 1]) + [0]*redundant
        submit['EMPIRICAL'] = list(prediction[:, 2]) + [0]*redundant
        submit['OTHERS'] = list(prediction[:, 3]) + [0]*redundant
    else:
        submit['THEORETICAL'] = [0]*redundant + list(prediction[:, 0])
        submit['ENGINEERING'] = [0]*redundant + list(prediction[:, 1])
        submit['EMPIRICAL'] = [0]*redundant + list(prediction[:, 2])
        submit['OTHERS'] = [0]*redundant + list(prediction[:, 3])
    df = pd.DataFrame.from_dict(submit)
    df.to_csv(filename, index=False)


def collect_words(df, n_workers=12):
    from multiprocessing import Pool
    from nltk.tokenize import word_tokenize

    sent_list = []
    for i in df.iterrows():
        sent_list += i[1]['Abstract'].split('$$$')

    chunks = [
        ' '.join(sent_list[i:i + len(sent_list) // n_workers])
        for i in range(0, len(sent_list), len(sent_list) // n_workers)
    ]
    with Pool(n_workers) as pool:
        chunks = pool.map_async(word_tokenize, chunks)
        words = set(sum(chunks.get(), []))

    return words

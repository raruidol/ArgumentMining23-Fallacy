from sentence_transformers import SentenceTransformer
from datasets import Dataset, DatasetDict
import json
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def load_dataset_multi_class(path):
    data = {'train': {}, 'dev': {}, 'test': {}}

    data['train']['label'] = []
    data['train']['text'] = []
    data['test']['label'] = []
    data['test']['text'] = []

    try:
        with open(path) as filehandle:
            json_data = json.load(filehandle)
    except:
        print('The file is not available.')
        exit()

    print('File loaded.')

    for sample in json_data['train']:
        data['train']['text'].append(sample[0])
        if sample[1] == 'None':
            data['train']['label'].append(0)
        elif sample[1] == 'AppealtoMajority':
            data['train']['label'].append(1)
        elif sample[1] == 'AppealtoAuthority':
            data['train']['label'].append(2)
        elif sample[1] == 'Slipperyslope':
            data['train']['label'].append(3)
        elif sample[1] == 'AdHominem':
            data['train']['label'].append(4)

    for sample in json_data['dev']:
        data['train']['text'].append(sample[0])
        if sample[1] == 'None':
            data['train']['label'].append(0)
        elif sample[1] == 'AppealtoMajority':
            data['train']['label'].append(1)
        elif sample[1] == 'AppealtoAuthority':
            data['train']['label'].append(2)
        elif sample[1] == 'Slipperyslope':
            data['train']['label'].append(3)
        elif sample[1] == 'AdHominem':
            data['train']['label'].append(4)

    for sample in json_data['test']:
        data['test']['text'].append(sample[0])
        if sample[1] == 'None':
            data['test']['label'].append(0)
        elif sample[1] == 'AppealtoMajority':
            data['test']['label'].append(1)
        elif sample[1] == 'AppealtoAuthority':
            data['test']['label'].append(2)
        elif sample[1] == 'Slipperyslope':
            data['test']['label'].append(3)
        elif sample[1] == 'AdHominem':
            data['test']['label'].append(4)

    final_data = DatasetDict()
    for k, v in data.items():
        final_data[k] = Dataset.from_dict(v)

    return final_data


def load_dataset_two_class(path):
    data = {'train': {}, 'dev': {}, 'test': {}}

    data['train']['label'] = []
    data['train']['text'] = []
    data['test']['label'] = []
    data['test']['text'] = []

    try:
        with open(path) as filehandle:
            json_data = json.load(filehandle)
    except:
        print('The file is not available.')
        exit()

    print('File loaded.')

    for sample in json_data['train']:
        data['train']['text'].append(sample[0])
        if sample[1] == 'None':
            data['train']['label'].append(0)
        else:
            data['train']['label'].append(1)

    for sample in json_data['dev']:
        data['train']['text'].append(sample[0])
        if sample[1] == 'None':
            data['train']['label'].append(0)
        else:
            data['train']['label'].append(1)

    for sample in json_data['test']:
        data['test']['text'].append(sample[0])
        if sample[1] == 'None':
            data['test']['label'].append(0)
        else:
            data['test']['label'].append(1)

    final_data = DatasetDict()
    for k, v in data.items():
        final_data[k] = Dataset.from_dict(v)

    return final_data


def load_dataset_fallacy_class(path):
    data = {'train': {}, 'dev': {}, 'test': {}}

    data['train']['label'] = []
    data['train']['text'] = []
    data['test']['label'] = []
    data['test']['text'] = []

    try:
        with open(path) as filehandle:
            json_data = json.load(filehandle)
    except:
        print('The file is not available.')
        exit()

    print('File loaded.')

    for sample in json_data['train']:

        if sample[1] == 'AdHominem':
            data['train']['text'].append(sample[0])
            data['train']['label'].append(0)
        elif sample[1] == 'AppealtoMajority':
            data['train']['text'].append(sample[0])
            data['train']['label'].append(1)
        elif sample[1] == 'AppealtoAuthority':
            data['train']['text'].append(sample[0])
            data['train']['label'].append(2)
        elif sample[1] == 'Slipperyslope':
            data['train']['text'].append(sample[0])
            data['train']['label'].append(3)

    for sample in json_data['dev']:

        if sample[1] == 'AdHominem':
            data['train']['text'].append(sample[0])
            data['train']['label'].append(0)
        elif sample[1] == 'AppealtoMajority':
            data['train']['text'].append(sample[0])
            data['train']['label'].append(1)
        elif sample[1] == 'AppealtoAuthority':
            data['train']['text'].append(sample[0])
            data['train']['label'].append(2)
        elif sample[1] == 'Slipperyslope':
            data['train']['text'].append(sample[0])
            data['train']['label'].append(3)

    for sample in json_data['test']:

        if sample[1] == 'AdHominem':
            data['test']['text'].append(sample[0])
            data['test']['label'].append(0)
        elif sample[1] == 'AppealtoMajority':
            data['test']['text'].append(sample[0])
            data['test']['label'].append(1)
        elif sample[1] == 'AppealtoAuthority':
            data['test']['text'].append(sample[0])
            data['test']['label'].append(2)
        elif sample[1] == 'Slipperyslope':
            data['test']['text'].append(sample[0])
            data['test']['label'].append(3)

    final_data = DatasetDict()
    for k, v in data.items():
        final_data[k] = Dataset.from_dict(v)

    return final_data


def vectorize_data(data, embedding):
    print('Generating embeddings...')

    tr_lb = data['train']['label']
    te_lb = data['test']['label']

    tr = embedding.encode(data['train']['text'])
    te = embedding.encode(data['test']['text'])

    print('Embeddings finished.')
    return np.array(tr),  np.array(te),  np.array(tr_lb),  np.array(te_lb)


def make_predictions(model, inputs):
    preds = model.predict(inputs)
    return preds


if __name__ == "__main__":

    data_path = 'data/fallacy_corpus.json'

    # LOAD DATA FOR THE MODEL
    dataset = load_dataset_fallacy_class(data_path)
    shuffled_dataset = dataset.shuffle(seed=42)

    embedding_model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')

    tr_x, te_x, tr_y, te_y = vectorize_data(shuffled_dataset, embedding_model)

    svc = make_pipeline(StandardScaler(), SVC(gamma='auto', C=1000))
    svc.fit(tr_x, tr_y)

    te_preds = make_predictions(svc, te_x)

    mf1_test = precision_recall_fscore_support(te_y, te_preds, average='macro')

    print('Macro F1 score in TEST:', mf1_test)
    print('Confusion matrix:')
    print(confusion_matrix(te_y, te_preds))

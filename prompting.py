import random
import openai
import json
from retry import retry
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

random.seed(4)

@retry()
def create_completion_5_class(text1):
    mode = 'ZERO-SHOT'
    cmpl = openai.ChatCompletion.create(
        model="gpt-4",
        # model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You task is to detect a fallacy in the Text Snippet. The label can be 'Slippery Slope', 'Appeal to Authority', 'Ad Hominem', 'Appeal to Majority' or 'None'."},
            {"role": "user", "content": text1},
            {"role": "user", "content": "Label: "}
        ]
    )
    return cmpl, mode


@retry()
def create_completion_fallacy_detection(text1):
    mode = 'ZERO-SHOT'
    cmpl = openai.ChatCompletion.create(
        model="gpt-4",
        # model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You task is to detect a fallacy in the Text Snippet. The label can be 'Fallacy' or 'None'."},
            {"role": "user", "content": text1},
            {"role": "user", "content": "Label: "}
        ]
    )
    return cmpl, mode


@retry()
def create_completion_fallacy_classification(text1):
    mode = 'ZERO-SHOT'
    cmpl = openai.ChatCompletion.create(
        model="gpt-4",
        #model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You task is to classify a fallacy in the Text Snippet. The label can be 'Slippery Slope', 'Appeal to Authority', 'Ad Hominem' or 'Appeal to Majority'."},
            {"role": "user", "content": text1},
            {"role": "user", "content": "Label: "}
        ]
    )
    return cmpl, mode


if __name__ == "__main__":
    CALLS = 0
    mode = 'None'

    # OPENAI API Key
    openai.api_key = ''

    # Evaluation vectors
    ground_truth = []
    gpt_preds = []

    with open('data/validation_corpus.json') as filehandle:
        json_data = json.load(filehandle)

    TOTAL_CALLS = len(json_data['test'])
    for sample in json_data['test']:

        # Text Snippet
        t1 = 'Text Snippet: '+sample[0]

        # Label
        if 'Slipperyslope' == sample[1]:
            ground_truth.append(1)
        elif 'AppealtoMajority' == sample[1]:
            ground_truth.append(2)
        elif 'AppealtoAuthority' == sample[1]:
            ground_truth.append(3)
        elif 'AdHominem' == sample[1]:
            ground_truth.append(4)
        else:
            ground_truth.append(0)

        # Create a completion and a prediction with GPT-Chat
        completion, mode = create_completion_fallacy_detection(t1)
        pred = completion.choices[0].message.content

        if 'slippery' in pred.lower():
            gpt_preds.append(1)
        elif 'majority' in pred.lower():
            gpt_preds.append(2)
        elif 'authority' in pred.lower():
            gpt_preds.append(3)
        elif 'hominem' in pred.lower():
            gpt_preds.append(4)
        else:
            gpt_preds.append(0)
        '''
        if 'fallacy' in pred.lower():
            gpt_preds.append(1)
        else:
            gpt_preds.append(0)
        '''
        CALLS += 1
        print(CALLS,'/',TOTAL_CALLS)
        print(pred)
        #print(gpt_preds[-1])

    mf1 = precision_recall_fscore_support(ground_truth, gpt_preds, average='macro')

    print(('******************************'))
    print(('******************************'))
    print(mode)
    print(('******************************'))
    print(('******************************'))
    print('Macro F1 score in TEST:', mf1)
    print(('******************************'))
    print('Confusion matrix')
    print(confusion_matrix(ground_truth, gpt_preds))
    print(('******************************'))
    print(('******************************'))
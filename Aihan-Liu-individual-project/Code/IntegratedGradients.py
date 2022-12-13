import re
import os
import numpy as np
import matplotlib as mpl
import matplotlib.cm

from tqdm import tqdm
from typing import Optional, Union, List, Dict, Tuple
from IPython.display import HTML
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from transformers.optimization_tf import WarmUp
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizer, AutoModelForSequenceClassification

from alibi.explainers import IntegratedGradients

def decode_sentence(x: List[int], reverse_index: Dict[int, str], unk_token: str = '[UNK]') -> str:
    """
    Decodes the tokenized sentences from keras IMDB dataset into plain text.

    Parameters
    ----------
    x
        List of integers to be docoded.
    revese_index
        Reverse index map, from `int` to `str`.
    unk_token
        Unkown token to be used.

    Returns
    -------
        Decoded sentence.
    """
    # the `-3` offset is due to the special tokens used by keras
    # see https://stackoverflow.com/questions/42821330/restore-original-text-from-keras-s-imdb-dataset
    return " ".join([reverse_index.get(i - 3, unk_token) for i in x])


def process_sentences(sentence: List[str],
                      tokenizer: PreTrainedTokenizer,
                      max_len: int) -> Dict[str, np.ndarray]:
    """
    Tokenize the text sentences.

    Parameters
    ----------
    sentence
        Sentence to be processed.
    tokenizer
        Tokenizer to be used.
    max_len
        Controls the maximum length to use by one of the truncation/padding parameters.


    Returns
    -------
    Tokenized representation containing:
     - input_ids
     - attention_mask
    """
    # since we are using the model for classification, we need to include special char (i.e, '[CLS]', ''[SEP]')
    # check the example here: https://huggingface.co/transformers/v4.4.2/quicktour.html
    z = tokenizer(sentence,
                  add_special_tokens=True,
                  padding='max_length',
                  max_length=max_len,
                  truncation=True,
                  return_attention_mask = True,
                  return_tensors='np')
    return z

def process_input(sentence: List[str],
                  tokenizer: PreTrainedTokenizer,
                  max_len: int) -> Tuple[np.ndarray, dict]:
    """
    Preprocess input sentence befor sending to transformer model.

    Parameters
    -----------
    sentence
        Sentence to be processed.
    tokenizer
        Tokenizer to be used.
    max_len
        Controls the maximum length to use by one of the truncation/padding parameters.

    Returns
    -------
    Tuple consisting of the input_ids and a dictionary contaning the attention_mask.
    """
    # tokenize the sentences using the transformer's tokenizer.
    tokenized_samples = process_sentences(sentence, tokenizer, max_len)
    X_test = tokenized_samples['input_ids'].astype(np.int32)

    # the values of the kwargs have to be `tf.Tensor`.
    # see transformers issue #14404: https://github.com/huggingface/transformers/issues/14404
    # solved from v4.16.0
    kwargs = {k: tf.constant(v) for k, v in tokenized_samples.items() if k != 'input_ids'}
    return X_test, kwargs


def  hlstr(string: str , color: str = 'white') -> str:
    """
    Return HTML markup highlighting text with the desired color.
    """
    return f"<mark style=background-color:{color}>{string} </mark>"


def colorize(attrs: np.ndarray, cmap: str = 'PiYG') -> List:
    """
    Compute hex colors based on the attributions for a single instance.
    Uses a diverging colorscale by default and normalizes and scales
    the colormap so that colors are consistent with the attributions.

    Parameters
    ----------
    attrs
        Attributions to be visualized.
    cmap
        Matplotlib cmap type.
    """
    cmap_bound = np.abs(attrs).max()
    norm = mpl.colors.Normalize(vmin=-cmap_bound, vmax=cmap_bound)
    cmap = mpl.cm.get_cmap(cmap)
    return list(map(lambda x: mpl.colors.rgb2hex(cmap(norm(x))), attrs))


def display(X: np.ndarray,
            attrs: np.ndarray,
            tokenizer: PreTrainedTokenizer,
            pred: np.ndarray) -> None:
    """
    Display the attribution of a given instance.

    Parameters
    ----------
    X
        Instance to display the attributions for.
    attrs
        Attributions values for the given instance.
    tokenizer
        Tokenizer to be used for decoding.
    pred
        Classification label (prediction) for the given instance.
    """
    # pred_dict = {1: 'Positive', 0: 'Negative'}
    pred_dict = {0: 'Negative', 1: 'Neutral', 2:"Positive"}

    # remove padding
    fst_pad_indices = np.where(X ==tokenizer.pad_token_id)[0]
    if len(fst_pad_indices) > 0:
        X, attrs = X[:fst_pad_indices[0]], attrs[:fst_pad_indices[0]]

    # decode tokens and get colors
    tokens = [tokenizer.decode([X[i]]) for i in range(len(X))][1:-1]
    colors = colorize(attrs)[1:-1]

    print(f'Predicted label =  {pred}: {pred_dict[pred]}')
    # html_results = HTML("".join(list(map(hlstr, tokens, colors))))
    html_text = "".join(list(map(hlstr, tokens, colors)))
    out_text = '<b>' + f'Predicted {pred_dict[pred]}' + r'<b>' + '&nbsp;&nbsp;&nbsp;&nbsp;' + html_text
    return out_text

class AutoModelWrapper(keras.Model):
    def __init__(self, transformer: keras.Model, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        transformer
            Transformer to be wrapped.
        """
        super().__init__()
        self.transformer = transformer

    def call(self,
             input_ids: Union[np.ndarray, tf.Tensor],
             attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
             training: bool = False):
        """
        Performs forward pass throguh the model.

        Parameters
        ----------
        input_ids
            Indices of input sequence tokens in the vocabulary.
        attention_mask
            Mask to avoid performing attention on padding token indices.

        Returns
        -------
        Classification probabilities.
        """
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask, training=training)
        return tf.nn.softmax(out.logits, axis=-1)

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

############################ imdb data example

# constants
max_features = 10000

# # load imdb reviews datasets.
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# remove the first integer token which is a special character that marks the beginning of the sentence
x_train = [x[1:] for x in x_train[0:100000, ]]
y_train = y_train[0:100000, ]
x_test = [x[1:] for x in x_test[0:100, ]]
y_test = y_test[0:100, ]



# get mappings. The keys are transformed to lower case since we will use uncased models.
reverse_index = {value: key.lower() for (key, value) in imdb.get_word_index().items()}

# choose whether to use the BERT or distilBERT model by selecting the appropriate name
# model_name = 'distilbert-base-uncased'
# model_name = 'bert-base-uncased'
# model_name = 'roberta-base'
model_name = "cardiffnlp/twitter-roberta-base-sentiment"

# load model and tokenizer
# model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# define maximum input length
max_len = 256

if model_name == 'bert-base-uncased':
    # training parameters: https://huggingface.co/fabriceyhc/bert-base-uncased-imdb
    init_lr = 5e-05
    min_lr_ratio = 0
    batch_size = 8
    num_warmup_steps = 10 # 1546
    num_train_steps = 10 # 15468
    power = 1.0

elif model_name == 'distilbert-base-uncased':
    # training parameters: https://huggingface.co/lvwerra/distilbert-imdb
    init_lr = 5e-05
    min_lr_ratio = 0
    batch_size = 16
    num_warmup_steps = 0
    num_train_steps = int(np.ceil(len(x_train) / batch_size))
    power = 1.0
elif model_name == 'roberta-base':
    init_lr = 5e-05
    min_lr_ratio = 0
    batch_size = 16
    num_warmup_steps = 0
    num_train_steps = int(np.ceil(len(x_train) / batch_size))
    power = 1.0

elif model_name == 'cardiffnlp/twitter-roberta-base-sentiment':
    # training parameters: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment
    init_lr = 5e-05
    min_lr_ratio = 0
    batch_size = 16
    num_warmup_steps = 0
    num_train_steps = int(np.ceil(len(x_train) / batch_size))
    power = 1.0
else:
    raise ValueError('Unknown model name.')


X_train, X_test = [], []

# decode training sentences
for i in range(len(x_train)):
    tr_sentence = decode_sentence(x_train[i], reverse_index, unk_token=tokenizer.unk_token)
    X_train.append(tr_sentence)

# decode testing sentences
for i in range(len(x_test)):
    te_sentence = decode_sentence(x_test[i], reverse_index, unk_token=tokenizer.unk_token)
    X_test.append(te_sentence)

# tokenize datasets
X_train = process_sentences(X_train, tokenizer, max_len)
X_test = process_sentences(X_test, tokenizer, max_len)

train_ds = tf.data.Dataset.from_tensor_slices(((*X_train.values() ,), y_train))
train_ds = train_ds.shuffle(1024).batch(batch_size).repeat()

test_ds = tf.data.Dataset.from_tensor_slices(((*X_test.values(), ), y_test))
test_ds = test_ds.batch(batch_size)


filepath = './Checkpoints/'  # change to desired save directory
checkpoint_path = os.path.join(filepath, model_name)
load_model = True
pretrained_model = True

# define linear learning schedules
lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=init_lr,
    decay_steps=num_train_steps - num_warmup_steps,
    end_learning_rate=init_lr * min_lr_ratio,
    power=power,
)

# include learning rate warmup
if num_warmup_steps:
    lr_schedule = WarmUp(
        initial_learning_rate=init_lr,
        decay_schedule_fn=lr_schedule,
        warmup_steps=num_warmup_steps,
    )

if not load_model:
    # compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=tf.metrics.SparseCategoricalAccuracy(),
    )

    # fit and save the model
    model.fit(x=train_ds, validation_data=test_ds, steps_per_epoch=num_train_steps)
    model.save_pretrained(checkpoint_path)
elif pretrained_model == True:
    model = model.from_pretrained(model_name)
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=tf.metrics.SparseCategoricalAccuracy(),
    )
    model.evaluate(test_ds)
else:
    # load and compile the model
    model = model.from_pretrained(checkpoint_path)
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=tf.metrics.SparseCategoricalAccuracy(),
    )
    # evaluate the model
    model.evaluate(test_ds)

# wrap the finetuned model
auto_model = AutoModelWrapper(model)

# include IMDB reviews from the test dataset
text_samples = [decode_sentence(x_test[i], reverse_index, unk_token=tokenizer.unk_token) for i in range(10)]

# inlcude your text here
text_samples.append("best movie i've ever seen nothing bad to say about it")

sub1 = pd.read_csv(r'../Data/MacOS_submission.csv', encoding='utf-8')
sub2 = pd.read_csv(r'../Data/windows_submission.csv', encoding='utf-8')

submission = pd.concat([sub1, sub2],ignore_index=True)

# submission['body'] = submission['title'].astype(str) + ' ' + submission['selftext'].astype(str)
submission['body'] = submission['title']

text_samples2 = submission['body'].tolist()

# process input before passing it to the explainer
X_test, kwargs = process_input(sentence=text_samples2[6:12] + text_samples2[-9:-3],
                               tokenizer=tokenizer,
                               max_len=max_len)

if model_name == 'bert-base-uncased':
    layer = auto_model.layers[0].layers[0].embeddings
    # layer = auto_model.layers[0].layers[0].encoder.layer[2]

# calculate the attributions with respect to the
# first embedding layer of the (distil)BERT
elif model_name == 'distilbert-base-uncased':
    layer = auto_model.layers[0].layers[0].embeddings
    # layer = auto_model.layers[0].layers[0].transformer.layer[0]

elif model_name == 'cardiffnlp/twitter-roberta-base-sentiment':
    layer = auto_model.layers[0].layers[0].embeddings
    # layer = auto_model.layers[0].layers[0].transformer.layer[0]

elif model_name == 'roberta-base':
    layer = auto_model.layers[0].layers[0].embeddings

else:
    raise ValueError('Unknown model name.')


n_steps = 50
method = "gausslegendre"
internal_batch_size = 5

# define Integrated Gradients explainer
ig = IntegratedGradients(auto_model,
                          layer=layer,
                          n_steps=n_steps,
                          method=method,
                          internal_batch_size=internal_batch_size)


# compute model's prediction and construct baselines
predictions = auto_model(X_test, **kwargs).numpy().argmax(axis=1)

# construct the baseline as before
mask = np.isin(X_test, tokenizer.all_special_ids)
baselines = X_test * mask + tokenizer.pad_token_id * (1 - mask)

# get explanation
explanation = ig.explain(X_test,
                         forward_kwargs=kwargs,
                         baselines=baselines,
                         target=predictions)

# Get attributions values from the explanation object
attrs = explanation.attributions[0]
print('Attributions shape:', attrs.shape)


attrs = attrs.sum(axis=2)
print('Attributions shape:', attrs.shape)

########### Check attributions for our example
res = []
for index in range(X_test.shape[0]):
    res1 = display(X=X_test[index], attrs=attrs[index], pred=predictions[index], tokenizer=tokenizer)
    res.append(res1)
html_str = '<br><br>'.join(res).encode('utf-8').strip()


# ########### Check attribution for some test examples
# index = 0
# res2 = display(X=X_test[index], attrs=attrs[index], pred=predictions[index], tokenizer=tokenizer)

# https://docs.seldon.io/projects/alibi/en/stable/examples/integrated_gradients_transformers.html

with open("Results/Results_trained.html", 'w', encoding='utf-8') as my_file:
    my_file.write(html_str.decode('utf-8'))
from Modules.model import RobertitoModel
from transformers import AutoTokenizer
from Modules.dataset import (
    create_dataloader,
    get_seq_and_mask,
    tokenize_data,
    get_weights,
)
from Modules.params import (
    get_params,
    get_args,
)
from pandas import read_csv

params = get_params()
args = get_args()
tokenizer = AutoTokenizer.from_pretrained(
    "pysentimiento/robertuito-base-uncased"
)
train_data = read_csv(
    params["train file"],
    sep="\t"
)
val_data = read_csv(
    params["val file"],
    sep="\t"
)
tokens_train = tokenize_data(
    train_data["Text"].to_list(),
    tokenizer,
    args
)
tokens_val = tokenize_data(
    val_data["Text"].to_list(),
    tokenizer,
    args
)
train_seq, train_mask, y_train = get_seq_and_mask(
    tokens_train,
    train_data
)
val_seq, val_mask, y_val = get_seq_and_mask(
    tokens_val,
    val_data
)
weights = get_weights(
    train_data,
    y_train,
    args
)
train_dataloader = create_dataloader(
    train_seq,
    train_mask,
    args,
    y_train
)
val_dataloader = create_dataloader(
    val_seq,
    val_mask,
    args,
    y_val
)
robertuito = RobertitoModel(
    params,
    args,
    weights
)
robertuito.run(
    train_dataloader,
    val_dataloader
)
# test_seq, test_mask = get_seq_and_mask(
# tokens_test
# )


# tokens_test = tokenize_data(
# test_data["text"].to_list(),
# tokenizer,
# args
# )

# test_dataloader = create_dataloader(test_seq,
# test_mask)

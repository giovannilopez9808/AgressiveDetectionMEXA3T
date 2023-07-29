from sklearn.utils import class_weight
from transformers import AutoTokenizer
from argparse import Namespace
from torch.utils.data import (
    TensorDataset,
    RandomSampler,
    DataLoader,
)
from pandas import DataFrame
from torch import (
    tensor,
    float,
)
from numpy import (
    unique,
    array,
)


def tokenize_data(
    data: list,
    tokenizer: AutoTokenizer,
    args: Namespace
) -> dict:
    """
    Realiza el proceso de tokenizacion de una lista de datos
    """
    token_list = tokenizer.batch_encode_plus(
        data,
        max_length=args.max_tokens,
        padding=True,
        add_special_tokens=True,
        truncation='longest_first')
    return token_list


def get_seq_and_mask(
    tokens_data: dict,
    data: DataFrame = None
) -> tuple:
    """
    Obtiene los tensores correspondientes a los datos introducidos.
    Si no se introduce un dataframe en la ultima capa esto indica que
    no existen los valores de target del dataset dado
    """
    seq = tensor(
        tokens_data['input_ids']
    )
    mask = tensor(
        tokens_data['attention_mask']
    )
    if data is not None:
        y = tensor(
            list(
                map(
                    int,
                    data["Label"].to_numpy()
                )
            )
        )
        return seq, mask, y
    return seq, mask


def get_weights(
    data: DataFrame,
    target: list,
    args: Namespace
) -> array:
    """
    Obtiene los pesos para realizar un balance en las
    categorias de los datos
    """
    # Pesos de cada clase
    weights_class = class_weight.compute_class_weight(
        'balanced',
        classes=unique(target),
        y=data["Label"].to_list()
    )
    # Conversion a tensor
    weights = tensor(
        weights_class,
        dtype=float
    )
    # LLevado hacia la GPU
    weights = weights.to(
        args.device
    )
    return weights


def create_dataloader(
    seq: array,
    mask: array,
    args: Namespace,
    target: array = None,
) -> tuple:
    """
    Crea los dataloaders de cada secuencia de datos
    """
    if target is not None:
        # Concatenacion de los datos en un solo tensor
        data = TensorDataset(
            seq,
            mask,
            target
        )
        # Muestras del tensor
        sampler = RandomSampler(
            data
        )
        # Creacion del dataloader en base a muestras
        dataloader = DataLoader(
            data,
            sampler=sampler,
            batch_size=args.batch_size
        )
        return dataloader

    # Espacio para la creacion del dataloader para los datos de test
    data = TensorDataset(
        seq,
        mask
    )
    dataloader = DataLoader(
        data,
        batch_size=args.batch_size
    )
    return dataloader

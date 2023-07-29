from argparse import Namespace
from os.path import join
from torch import device


def get_params() -> dict:
    """
    Direcciones y nombres de los archivos donde estan contenidos 
    los datos que se usaran para el entrenamiento, validacion y test
    """
    params = {
        "path data": "../Data",
        "path results": "../Results",
        "train file": "train.csv",
        "val file": "val.csv",
        "test file": "test.csv",
    }
    params["train file"] = join(
        params["path data"],
        params["train file"]
    )
    params["val file"] = join(
        params["path data"],
        params["val file"]
    )
    params["test file"] = join(
        params["path data"],
        params["test file"]
    )
    return params


def get_args() -> Namespace:
    """
    Argumentos utilizados para el tokenizador y el proceso de entrenamiento
    de la red neuronal
    """
    args = Namespace()
    args.max_tokens = 130
    args.batch_size = 8
    args.epoch = 10
    args.lr = 1e-5
    args.device = device(
        'cuda'
    )
    return args

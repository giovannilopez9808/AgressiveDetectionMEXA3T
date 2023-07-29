from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score
from torch.nn.utils import clip_grad_norm_
from torch.nn.functional import (
    log_softmax,
    softmax
)
from argparse import Namespace
from torch.optim import AdamW
from pandas import DataFrame
from numpy import array
from torch.nn import (
    CrossEntropyLoss,
    Module,
)
from torch import (
    save,
    load,
)
from tqdm import tqdm


class RobertuitoClasificator(Module):
    """
    Creacion del modelo de Robertuito
    """

    def __init__(
            self,
            transformer
    ) -> None:
        super(
            RobertuitoClasificator,
            self
        ).__init__()
        # pretrained
        self.transformer = transformer

    def forward(
            self,
            sent_id,
            mask
    ) -> array:
        # Get cls token
        x = self.transformer(
            sent_id,
            attention_mask=mask,
            return_dict=False
        )
        x = x[0]
        return x


class RobertitoModel:
    """
    Aglomeracion de los procesos de entrenamiento, validacion y test del modelo
    de robertito
    """

    def __init__(
        self, params: dict,
        args: Namespace,
        weights: array
    ) -> None:
        self.params = params
        self.args = args
        self._create_model(weights)

    def _create_model(
        self,
        weights: array
    ) -> None:
        """
        Definicion del modelo, optimizador y funcion de costo
        """
        robertuito = AutoModelForSequenceClassification.from_pretrained(
            "pysentimiento/robertuito-base-uncased"
        )
        robertuito = robertuito.to(
            self.args.device
        )
        # Creacion del modelo
        self.model = RobertuitoClasificator(
            robertuito
        )
        # Llevado al GPU
        self.model = self.model.to(
            self.args.device
        )
        # Optimizador
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.lr
        )
        # Funcion de costo
        self.cross_entropy = CrossEntropyLoss(
            weight=weights
        )

    def _train(
            self,
            train_dataloader
    ):
        """
        Entrenamieto del 
        """
        self.model.train()
        # Inicializacion de las variables
        total_loss = 0
        acc = 0
        predictions = []
        targets = []
        for step, batch in tqdm(
            enumerate(train_dataloader),
            desc="Training",
            total=len(train_dataloader)
        ):
            # Datos hacia la GPU
            batch = [
                r.to(self.args.device)
                for r in batch
            ]
            sent_id, mask, labels = batch
            # Limpieza de procesos anteriores
            self.model.zero_grad()
            # Prediccion
            preds = self.model(
                sent_id,
                mask
            )
            lab = log_softmax(
                preds,
                dim=1
            )
            # Funcion de costo
            loss = self.cross_entropy(
                lab,
                labels
            )
            total_loss = total_loss + loss.item()
            loss.backward()
            # Maximo de 1 para que no explote
            clip_grad_norm_(
                self.model.parameters(),
                1.0
            )
            # Actualizacion de los parametrod
            self.optimizer.step()
            # Predicciones a cpu
            preds = preds.detach().cpu().numpy()
            # Guardado de las preducciones
            predictions += lab.argmax(1).cpu().tolist()
            targets += labels.cpu().tolist()
        # Accuracy
        acc = accuracy_score(
            targets,
            predictions
        )
        avg_loss = total_loss / len(train_dataloader)
        return avg_loss, acc

    def _evaluate(
        self,
        val_dataloader
    ):
        """
        Evaluacion del modelo
        """
        # Desactivacion del entrenamiento
        self.model.eval()
        # Inicializacion de las variables
        total_loss = 0
        total_preds = []
        targets = []
        predictions = []
        for step, batch in tqdm(
            enumerate(val_dataloader),
            desc="Validation",
            total=len(val_dataloader)
        ):
            # Llevado hacia la GPU
            batch = [
                t.to(self.args.device)
                for t in batch
            ]
            sent_id, mask, labels = batch
            # No toma en cuenta el gradiente
            self.model.eval()
            # Predicciones
            preds = self.model(
                sent_id,
                mask
            )
            lab = log_softmax(
                preds,
                dim=1
            ).argmax(1)
            # Funcion de costo
            loss = self.cross_entropy(
                preds,
                labels
            )
            total_loss = total_loss + loss.item()
            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)
            predictions += lab.cpu().tolist()
            targets += labels.cpu().tolist()
        acc = accuracy_score(
            targets,
            predictions
        )
        # Funcion de costo promedio
        avg_loss = total_loss / len(val_dataloader)
        return avg_loss, acc

    def run(
        self,
        train_dataloader,
        val_dataloader
    ) -> None:
        """
        Ejecuccion del entrenamiento del modelo y su validacion
        """
        self.train = dict(
            loss=[],
            acc=[]
        )
        self.val = dict(
            loss=[],
            acc=[]
        )
        for epoch in range(self.args.epoch):
            print("="*30)
            print(f"Epoch {epoch+1}")
            # Entrenamiento dle modelo
            train_loss, train_acc = self._train(
                train_dataloader
            )
            self.train["loss"] += [train_loss]
            self.train["acc"] += [train_acc]
            # Evaluacion del modelo
            val_loss, val_acc = self._evaluate(
                val_dataloader
            )
            self.val["loss"] += [val_loss]
            self.val["acc"] += [val_acc]
            # Guardado de cada modelo en cada epoca
            save(
                self.model.state_dict(),
                f'saved_weights{epoch}.pt'
            )
            print(
                'Training Loss:   {:.4f}\t Training accuracy:  {:.4f}'.format(
                    train_loss,
                    train_acc
                )
            )
            print(
                'Validation Loss: {:.4f}\t Validation accuracy {:.4f}'.format(
                    val_loss,
                    val_acc
                )
            )

    def load(
        self,
        path: str
    ) -> None:
        """
        Lectura de algun estado del modelo
        """
        self.model.load_state_dict(
            load(
                path
            )
        )
        self.model.eval()

    def predict(
            self,
            test_dataloader
    ):
        """
        Prediccion dado un dataloader
        """
        predictions = []
        for batch in test_dataloader:
            sent_id, masks = batch
            sent_id = sent_id.to(
                self.args.device
            )
            # LLevado hacia la GPU
            masks = masks.to(
                self.args.device
            )
            # Predicciones
            output = self.model(sent_id,
                                masks)
            # Probabilidad
            preds = softmax(
                output,
                dim=1).argmax(1)
            # Guardado de las predicciones
            predictions += preds.tolist()
        # LLevado a un dataframe
        predictions = DataFrame(
            predictions,
            columns=["Expected"]
        )
        predictions.index.name = "Id"
        return predictions

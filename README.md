# Deteccion de tweets agresivos

Modelo de clasificación de tweets agresivos y no agresivos. El modelo tiene como base el modelo [RoBERTuito](https://github.com/pysentimiento/robertuito), el modelo fue entrenado bajo los esquemas de full y fine tunning con base en los datos de la competencia [MEXA3T](https://sites.google.com/view/mex-a3t/home).

### Entrenamiento y evaluacion del modelo
```bash
python train.py
```


### Requisitos
```bash
transformers==4.31.0
sklearn==1.2.2
numpy==1.25.1
pandas==2.0.1
torch==2.0.1
tqdm==4.65.0
```

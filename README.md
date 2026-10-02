# cnnbiome

- Convolutional Neural Network on microbiome
- fastai approach
- Adaptive pooling for variable length

```
Train:
    python dna_cnn_cli.py train \\
        --data sequences.csv \\
        --seq-col sequences --label-col label \\
        --epochs 5 --lr 1e-3 --bs 64 \\
        --model-out dna_cnn_model.pth

Predict:
    python dna_cnn_cli.py predict \\
        --model dna_cnn_model.pth \\
        --data new_sequences.csv --seq-col sequences \\
        --out predictions.csv

```


Gaurav Sablok \
gsablok@proton.me
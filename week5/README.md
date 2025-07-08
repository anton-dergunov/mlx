
## Audio Classification

https://huggingface.co/datasets/danavery/urbansound8K

What is VGG-like, CRNN?
Cosine Annealing or ReduceLROnPlateau

Audio augmentations:
- Time shifting
- Adding background noise
- Pitch shifting
- SpecAugment (mask time/frequency bands on the spectrogram)

References for inspiration
- Piczak CNN baseline — simple CNN that works well.
  https://github.com/karoldvl/ESC-50/blob/master/src/models.py
- PANNs — powerful pretrained audio CNNs.
  https://github.com/qiuqiangkong/audioset_tagging_cnn
- AST paper — Audio Spectrogram Transformer.
  https://arxiv.org/abs/2104.01778
- torchaudio tutorials.
  https://pytorch.org/audio/stable/tutorials/index.html

Using salience?

- Display spectrogram with moving head to visualize
https://stackoverflow.com/questions/59641390/jupyter-widget-to-play-audio-with-playhead-on-graph
https://gist.github.com/deeplycloudy/2152643
https://github.com/scottire/fastpages/blob/master/_notebooks/2020-10-21-interactive-audio-plots-in-jupyter-notebook.ipynb

Overfitting displayed as %-tage?
"The percentage difference between average loss over validation set and average lost over last training epoch"
Is this a legit metric?

Results for CNN:

```
Model architecture:
 SimpleCNN(
  (conv1): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (dropout): Dropout(p=0.3, inplace=False)
  (fc1): Linear(in_features=10752, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)
Total parameters: 1400970

=== Fold 1 ===
Epoch 1/10 | Loss: 1.4876
Epoch 2/10 | Loss: 0.8760
Epoch 3/10 | Loss: 0.6216
Epoch 4/10 | Loss: 0.4733
Epoch 5/10 | Loss: 0.3424
Epoch 6/10 | Loss: 0.2846
Epoch 7/10 | Loss: 0.2214
Epoch 8/10 | Loss: 0.1932
Epoch 9/10 | Loss: 0.1539
Epoch 10/10 | Loss: 0.1598
Fold 1 | Accuracy: 0.7056 | Macro F1: 0.7313

=== Fold 2 ===
Epoch 1/10 | Loss: 0.2619
Epoch 2/10 | Loss: 0.1558
Epoch 3/10 | Loss: 0.1486
Epoch 4/10 | Loss: 0.1252
Epoch 5/10 | Loss: 0.1116
Epoch 6/10 | Loss: 0.1128
Epoch 7/10 | Loss: 0.0898
Epoch 8/10 | Loss: 0.0774
Epoch 9/10 | Loss: 0.0686
Epoch 10/10 | Loss: 0.1111
Fold 2 | Accuracy: 0.8446 | Macro F1: 0.8565

=== Fold 3 ===
Epoch 1/10 | Loss: 0.1259
Epoch 2/10 | Loss: 0.0901
Epoch 3/10 | Loss: 0.0894
Epoch 4/10 | Loss: 0.0739
Epoch 5/10 | Loss: 0.0621
Epoch 6/10 | Loss: 0.0785
Epoch 7/10 | Loss: 0.1148
Epoch 8/10 | Loss: 0.0934
Epoch 9/10 | Loss: 0.0543
Epoch 10/10 | Loss: 0.0581
Fold 3 | Accuracy: 0.9600 | Macro F1: 0.9617

=== Fold 4 ===
Epoch 1/10 | Loss: 0.0856
Epoch 2/10 | Loss: 0.0895
Epoch 3/10 | Loss: 0.0674
Epoch 4/10 | Loss: 0.0644
Epoch 5/10 | Loss: 0.0522
Epoch 6/10 | Loss: 0.0650
Epoch 7/10 | Loss: 0.0755
Epoch 8/10 | Loss: 0.0801
Epoch 9/10 | Loss: 0.0501
Epoch 10/10 | Loss: 0.0766
Fold 4 | Accuracy: 0.9667 | Macro F1: 0.9659

=== Fold 5 ===
Epoch 1/10 | Loss: 0.0937
Epoch 2/10 | Loss: 0.0651
Epoch 3/10 | Loss: 0.0500
Epoch 4/10 | Loss: 0.0603
Epoch 5/10 | Loss: 0.0542
Epoch 6/10 | Loss: 0.0813
Epoch 7/10 | Loss: 0.0681
Epoch 8/10 | Loss: 0.0560
Epoch 9/10 | Loss: 0.0457
Epoch 10/10 | Loss: 0.0761
Fold 5 | Accuracy: 0.9776 | Macro F1: 0.9780

=== Fold 6 ===
Epoch 1/10 | Loss: 0.0969
Epoch 2/10 | Loss: 0.0771
Epoch 3/10 | Loss: 0.0433
Epoch 4/10 | Loss: 0.0575
Epoch 5/10 | Loss: 0.0713
Epoch 6/10 | Loss: 0.0958
Epoch 7/10 | Loss: 0.0448
Epoch 8/10 | Loss: 0.0518
Epoch 9/10 | Loss: 0.0511
Epoch 10/10 | Loss: 0.0473
Fold 6 | Accuracy: 0.9830 | Macro F1: 0.9847

=== Fold 7 ===
Epoch 1/10 | Loss: 0.0686
Epoch 2/10 | Loss: 0.0752
Epoch 3/10 | Loss: 0.0900
Epoch 4/10 | Loss: 0.0455
Epoch 5/10 | Loss: 0.0395
Epoch 6/10 | Loss: 0.0396
Epoch 7/10 | Loss: 0.0636
Epoch 8/10 | Loss: 0.0623
Epoch 9/10 | Loss: 0.0520
Epoch 10/10 | Loss: 0.0961
Fold 7 | Accuracy: 0.9523 | Macro F1: 0.9538

=== Fold 8 ===
Epoch 1/10 | Loss: 0.0793
Epoch 2/10 | Loss: 0.0721
Epoch 3/10 | Loss: 0.0460
Epoch 4/10 | Loss: 0.0790
Epoch 5/10 | Loss: 0.0406
Epoch 6/10 | Loss: 0.0548
Epoch 7/10 | Loss: 0.0645
Epoch 8/10 | Loss: 0.0527
Epoch 9/10 | Loss: 0.0486
Epoch 10/10 | Loss: 0.0586
Fold 8 | Accuracy: 0.9839 | Macro F1: 0.9850

=== Fold 9 ===
Epoch 1/10 | Loss: 0.0506
Epoch 2/10 | Loss: 0.0707
Epoch 3/10 | Loss: 0.0612
Epoch 4/10 | Loss: 0.0704
Epoch 5/10 | Loss: 0.0512
Epoch 6/10 | Loss: 0.0306
Epoch 7/10 | Loss: 0.0754
Epoch 8/10 | Loss: 0.1383
Epoch 9/10 | Loss: 0.0341
Epoch 10/10 | Loss: 0.0264
Fold 9 | Accuracy: 0.9853 | Macro F1: 0.9871

=== Fold 10 ===
Epoch 1/10 | Loss: 0.0683
Epoch 2/10 | Loss: 0.0958
Epoch 3/10 | Loss: 0.0373
Epoch 4/10 | Loss: 0.0948
Epoch 5/10 | Loss: 0.0709
Epoch 6/10 | Loss: 0.1008
Epoch 7/10 | Loss: 0.0630
Epoch 8/10 | Loss: 0.0280
Epoch 9/10 | Loss: 0.0381
Epoch 10/10 | Loss: 0.0523
Fold 10 | Accuracy: 0.9952 | Macro F1: 0.9959

=== 10-fold CV Results ===
Mean Accuracy: 0.9354 ± 0.0868
Mean Macro F1: 0.9400 ± 0.0792
```



## Audio Classification

https://github.com/openai/whisper

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

Results for Transformer:

```
Model architecture:
 AudioTransformer(
  (conv): Sequential(
    (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
    (6): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU()
  )
  (proj): Linear(in_features=1024, out_features=256, bias=True)
  (pos_encoder): PositionalEncoding()
  (transformer_encoder): TransformerEncoder(
    (layers): ModuleList(
      (0-3): 4 x TransformerEncoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
        )
        (linear1): Linear(in_features=256, out_features=512, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=512, out_features=256, bias=True)
        (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
      )
    )
  )
  (classifier): Linear(in_features=256, out_features=10, bias=True)
)
Total parameters: 2466506

=== Fold 1 ===
Epoch 1/10 | Loss: 1.6922
Epoch 2/10 | Loss: 0.8898
Epoch 3/10 | Loss: 0.6264
Epoch 4/10 | Loss: 0.4279
Epoch 5/10 | Loss: 0.2903
Epoch 6/10 | Loss: 0.2542
Epoch 7/10 | Loss: 0.2351
Epoch 8/10 | Loss: 0.1882
Epoch 9/10 | Loss: 0.1624
Epoch 10/10 | Loss: 0.1671
Fold 1 | Accuracy: 0.6770 | Macro F1: 0.7011

=== Fold 2 ===
Epoch 1/10 | Loss: 0.2414
Epoch 2/10 | Loss: 0.1443
Epoch 3/10 | Loss: 0.1484
Epoch 4/10 | Loss: 0.1215
Epoch 5/10 | Loss: 0.1094
Epoch 6/10 | Loss: 0.1378
Epoch 7/10 | Loss: 0.0846
Epoch 8/10 | Loss: 0.0901
Epoch 9/10 | Loss: 0.0895
Epoch 10/10 | Loss: 0.1188
Fold 2 | Accuracy: 0.7894 | Macro F1: 0.8054

=== Fold 3 ===
Epoch 1/10 | Loss: 0.1408
Epoch 2/10 | Loss: 0.1145
Epoch 3/10 | Loss: 0.0751
Epoch 4/10 | Loss: 0.0829
Epoch 5/10 | Loss: 0.0787
Epoch 6/10 | Loss: 0.1021
Epoch 7/10 | Loss: 0.0914
Epoch 8/10 | Loss: 0.0530
Epoch 9/10 | Loss: 0.0642
Epoch 10/10 | Loss: 0.0773
Fold 3 | Accuracy: 0.8605 | Macro F1: 0.8741

=== Fold 4 ===
Epoch 1/10 | Loss: 0.1130
Epoch 2/10 | Loss: 0.0748
Epoch 3/10 | Loss: 0.0845
Epoch 4/10 | Loss: 0.1213
Epoch 5/10 | Loss: 0.0851
Epoch 6/10 | Loss: 0.1139
Epoch 7/10 | Loss: 0.1297
Epoch 8/10 | Loss: 0.1012
Epoch 9/10 | Loss: 0.0994
Epoch 10/10 | Loss: 0.0857
Fold 4 | Accuracy: 0.9040 | Macro F1: 0.9013

=== Fold 5 ===
Epoch 1/10 | Loss: 0.1266
Epoch 2/10 | Loss: 0.0842
Epoch 3/10 | Loss: 0.0894
Epoch 4/10 | Loss: 0.0866
Epoch 5/10 | Loss: 0.0811
Epoch 6/10 | Loss: 0.0752
Epoch 7/10 | Loss: 0.1087
Epoch 8/10 | Loss: 0.0863
Epoch 9/10 | Loss: 0.1559
Epoch 10/10 | Loss: 0.1392
Fold 5 | Accuracy: 0.9252 | Macro F1: 0.9259

=== Fold 6 ===
Epoch 1/10 | Loss: 0.1313
Epoch 2/10 | Loss: 0.0650
Epoch 3/10 | Loss: 0.0644
Epoch 4/10 | Loss: 0.1100
Epoch 5/10 | Loss: 0.0859
Epoch 6/10 | Loss: 0.1661
Epoch 7/10 | Loss: 0.1330
Epoch 8/10 | Loss: 0.1380
Epoch 9/10 | Loss: 0.1051
Epoch 10/10 | Loss: 0.1439
Fold 6 | Accuracy: 0.7679 | Macro F1: 0.7755

=== Fold 7 ===
Epoch 1/10 | Loss: 0.1700
Epoch 2/10 | Loss: 0.0792
Epoch 3/10 | Loss: 0.0773
Epoch 4/10 | Loss: 0.1150
Epoch 5/10 | Loss: 0.0902
Epoch 6/10 | Loss: 0.1032
Epoch 7/10 | Loss: 0.0753
Epoch 8/10 | Loss: 0.0975
Epoch 9/10 | Loss: 0.0942
Epoch 10/10 | Loss: 0.0765
Fold 7 | Accuracy: 0.9236 | Macro F1: 0.9274

=== Fold 8 ===
Epoch 1/10 | Loss: 0.1040
Epoch 2/10 | Loss: 0.1094
Epoch 3/10 | Loss: 0.0841
Epoch 4/10 | Loss: 0.0554
Epoch 5/10 | Loss: 0.0714
Epoch 6/10 | Loss: 0.0718
Epoch 7/10 | Loss: 0.0571
Epoch 8/10 | Loss: 0.0737
Epoch 9/10 | Loss: 0.0578
Epoch 10/10 | Loss: 0.0770
Fold 8 | Accuracy: 0.9454 | Macro F1: 0.9537

=== Fold 9 ===
Epoch 1/10 | Loss: 0.1035
Epoch 2/10 | Loss: 0.0619
Epoch 3/10 | Loss: 0.0677
Epoch 4/10 | Loss: 0.1277
Epoch 5/10 | Loss: 0.1050
Epoch 6/10 | Loss: 0.0691
Epoch 7/10 | Loss: 0.0782
Epoch 8/10 | Loss: 0.0746
Epoch 9/10 | Loss: 0.0930
Epoch 10/10 | Loss: 0.0732
Fold 9 | Accuracy: 0.9498 | Macro F1: 0.9553

=== Fold 10 ===
Epoch 1/10 | Loss: 0.1028
Epoch 2/10 | Loss: 0.0806
Epoch 3/10 | Loss: 0.0694
Epoch 4/10 | Loss: 0.0602
Epoch 5/10 | Loss: 0.0772
Epoch 6/10 | Loss: 0.0781
Epoch 7/10 | Loss: 0.1357
Epoch 8/10 | Loss: 0.1297
Epoch 9/10 | Loss: 0.0667
Epoch 10/10 | Loss: 0.0703
Fold 10 | Accuracy: 0.9128 | Macro F1: 0.9196

=== 10-fold CV Results ===
Mean Accuracy: 0.8656 ± 0.0866
Mean Macro F1: 0.8739 ± 0.0810
```


## Emotion Classifier

https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio



## Mandarin Chinese Pronunciation Trainer

Main inspiration:
https://github.com/lars76/forced-alignment-chinese?tab=readme-ov-file

https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html
https://en.data-baker.com/datasets/freeDatasets/
https://sos1sos2sixteen.github.io/aishell3/
https://sos1sos2sixteen.github.io/aishell3/index.html
https://arxiv.org/abs/2010.11567
https://github.com/coqui-ai/TTS
https://github.com/readbeyond/aeneas
https://docs.pytorch.org/audio/main/tutorials/forced_alignment_tutorial.html
https://github.com/m-bain/whisperX
https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn
https://github.com/nazdridoy/kokoro-tts
https://github.com/suno-ai/bark
https://www.reddit.com/r/LocalLLaMA/comments/1dtzfte/best_tts_model_right_now_that_i_can_self_host/
https://mim-armand.medium.com/montreal-forced-alignment-hebrew-de7bd4180d6a

Other references:
https://wanghelin1997.github.io/CapSpeech-demo/


## Speaker recognition (diarization)

https://www.robots.ox.ac.uk/~vgg/data/voxconverse/index.html
https://huggingface.co/datasets/diarizers-community/voxconverse
https://umotion.univ-lemans.fr/video/9513-speech-segmentation-and-speaker-diarization/
https://gist.github.com/hbredin/049f2b629700bcea71324d2c1e7f8337
https://github.com/yinruiqing/pyannote-whisper
https://gist.github.com/alunkingusw/2eb29682a98f94a714d10080ed0f4896
https://github.com/openai/whisper/discussions/2609


## Notes

Use the dataset for corals from Ben
Crema-D dataset for emotions. Visualize MEL spectrograms for different emotions
Add emotion token to the output of Whisper such as <|emotion-happy|>
Mamba
  Efficiently modeling long sequences with structured state spaces
  Selective state space model
CNN for music genre recognition
  CDDataset
Weak supervisio in "Robust Speech recognition" paper
Voice2midi encoder decoder
DEMUCs for vocal extraction (from Facebook)
Style transfer (with Whisper?)
Input SATB, output different vocals
  ConvTasNet for sound seperation
Am I using hidden state from encoder part of Whisper or from decoder in Mandarin trainer app?
Attention pooling instead of avg pooling
HF (huggingface) trainer
HF datacollator
HF steps
TODO Read huggingface tutorials, their way of doing things.
QWen2-Audio
QWen2-Audio-Instruct
exploding gradients with CNN:
- clamping tensors
- clipping gradients
- batch norm
- reducing learning rate
- using gelu/selu instead of relu
- logging the mel-spectrogram (this worked)

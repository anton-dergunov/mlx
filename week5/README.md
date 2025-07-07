
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


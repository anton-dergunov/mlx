
- Make position embeddings optional (and estimate the impact)
- Use average pooling instead of CLS (and estimate the impact)
- Evaluate using patches of other size
- Slightly rotate the images in random mode

- Create a notebook showing all operations of the Visual Transformer step by step

- Visualize attention maps and do other exploration of the internals
- Print misclassified images

- Profile the model and understand which parts are slow (it has become slower after integrating VisionToSequence)

Fixes:
- Make sure that data loaders are deterministic

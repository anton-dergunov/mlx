import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Config
fig, ax = plt.subplots(figsize=(12, 2))

# Token dimensions
box_width = 1
box_height = 1
gap = 0.1

# Total positions
image_tokens = 5
caption_tokens = 8  # Example caption length incl. BOS/EOS
pad_tokens = 2

# Draw image tokens
for i in range(image_tokens):
    rect = patches.Rectangle(
        (i*(box_width+gap), 1.5),
        box_width,
        box_height,
        linewidth=1,
        edgecolor='black',
        facecolor='#8FBC8F'  # greenish
    )
    ax.add_patch(rect)

# Draw arch for image tokens
ax.annotate(
    'projected image embeddings (input)',
    xy=(image_tokens*(box_width+gap)/2 - box_width/2, 2.6),
    xytext=(image_tokens*(box_width+gap)/2 - box_width/2, 3.2),
    ha='center',
    arrowprops=dict(arrowstyle='-|>', color='gray')
)

# Draw caption tokens
caption_start = image_tokens*(box_width+gap) + gap*3
for i in range(caption_tokens):
    rect = patches.Rectangle(
        (caption_start + i*(box_width+gap), 1.5),
        box_width,
        box_height,
        linewidth=1,
        edgecolor='black',
        facecolor='#ADD8E6'  # pale blue
    )
    ax.add_patch(rect)
    if i == 0:
        ax.text(caption_start + i*(box_width+gap) + box_width/2, 2, 'BOS', ha='center', va='center', fontsize=8)
    elif i == caption_tokens-1:
        ax.text(caption_start + i*(box_width+gap) + box_width/2, 2, 'EOS', ha='center', va='center', fontsize=8)

# Draw arch for caption tokens
ax.annotate(
    'image caption (generated)',
    xy=(caption_start + caption_tokens*(box_width+gap)/2 - box_width/2, 0.4),
    xytext=(caption_start + caption_tokens*(box_width+gap)/2 - box_width/2, -0.2),
    ha='center',
    arrowprops=dict(arrowstyle='-|>', color='gray')
)

# Draw pad tokens
pad_start = caption_start + caption_tokens*(box_width+gap) + gap*3
for i in range(pad_tokens):
    rect = patches.Rectangle(
        (pad_start + i*(box_width+gap), 1.5),
        box_width,
        box_height,
        linewidth=1,
        edgecolor='black',
        facecolor='lightgray'
    )
    ax.add_patch(rect)
    ax.text(pad_start + i*(box_width+gap) + box_width/2, 2, 'PAD', ha='center', va='center', fontsize=8)

# Clean up
ax.axis('off')
ax.set_xlim(-1, pad_start + pad_tokens*(box_width+gap) + 1)
ax.set_ylim(0, 4)

plt.tight_layout()
plt.show()
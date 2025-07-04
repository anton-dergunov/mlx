## CLIP + 1 layer MLP (trained) + DistilGPT2 (trained)

=== Example ===
Reference: A dark-skinned woman, wearing a pink top and brown and white skirt, is crossing the street.
Generated: A woman in a red dress and a girl in a pink dress are walking down a street.

=== Example ===
Reference: Two soldiers are going into a tent with men and women watching them.
Generated: A group of people are gathered around a tent.

=== Example ===
Reference: A young girl in a blue shirt is bending over playing with a pile of sand on what seems to be a beach.
Generated: A young girl is digging a large pile of sand.

=== Example ===
Reference: There is a guy wearing stilts playing the trombone in front of a mural.
Generated: A woman in a white dress is standing on a platform with a large instrument.

=== Example ===
Reference: A man wearing a black sweater cooks food in a pan while standing in a cluttered kitchen.
Generated: A man in a green shirt is cooking.

Validation loss: 1.2562 | Average BLEU: 0.0761
Model saved to /root/models/model_20250703_145545_local_final.pt


## CLIP + 1 layer MLP (trained) + DistilGPT2 (frozen)

=== Example ===
Reference: A dark-skinned woman, wearing a pink top and brown and white skirt, is crossing the street.
Generated: A woman in a white dress is walking along a street.

=== Example ===
Reference: Two soldiers are going into a tent with men and women watching them.
Generated: A group of men are standing in a tent.

=== Example ===
Reference: A young girl in a blue shirt is bending over playing with a pile of sand on what seems to be a beach.
Generated: A little girl is playing with sand on the beach.

=== Example ===
Reference: There is a guy wearing stilts playing the trombone in front of a mural.
Generated: A woman in a black suit is standing on a stage.

=== Example ===
Reference: A man wearing a black sweater cooks food in a pan while standing in a cluttered kitchen.
Generated: A man in a white shirt is cooking a dish.

Validation loss: 1.2239 | Average BLEU: 0.0579


## CLIP + 2 layer MLP (trained) + DistilGPT2 (trained)

=== Example ===
Reference: A dark-skinned woman, wearing a pink top and brown and white skirt, is crossing the street.
Generated: A woman in a dress is walking with a young girl.

=== Example ===
Reference: Two soldiers are going into a tent with men and women watching them.
Generated: A group of women and children stand in front of a tent.

=== Example ===
Reference: A young girl in a blue shirt is bending over playing with a pile of sand on what seems to be a beach.
Generated: A little girl in a blue shirt is digging in the sand.

=== Example ===
Reference: There is a guy wearing stilts playing the trombone in front of a mural.
Generated: A woman in a black shirt and jeans is standing on a sidewalk.

=== Example ===
Reference: A man wearing a black sweater cooks food in a pan while standing in a cluttered kitchen.
Generated: A man in a black shirt is cooking.

Validation loss: 0.9760 | Average BLEU: 0.0753


## CLIP + 1 layer MLP (trained) + Qwen (trained)

In progress



## CLIP + 1 layer MLP (trained) + Custom Decoder (trained)

=== Example ===
Reference: A dark-skinned woman, wearing a pink top and brown and white skirt, is crossing the street.
Generated: A woman in a white dress and red skirt is walking down the street.

=== Example ===
Reference: Two soldiers are going into a tent with men and women watching them.
Generated: A tent with people standing around.

=== Example ===
Reference: A young girl in a blue shirt is bending over playing with a pile of sand on what seems to be a beach.
Generated: A young girl in a blue shirt is playing in the sand at the beach.

=== Example ===
Reference: There is a guy wearing stilts playing the trombone in front of a mural.
Generated: A woman in a white shirt and black pants is holding a cellphone.

=== Example ===
Reference: A man wearing a black sweater cooks food in a pan while standing in a cluttered kitchen.
Generated: A man in a black shirt is holding a pot of food.

Validation loss: 1.1380 | Average BLEU: 0.0596


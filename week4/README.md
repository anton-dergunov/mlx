
- TODO Investigate model collapse

```
Training: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 7754/7754 [12:49<00:00, 10.07it/s]
Train Loss: 1.2815
Model saved to /root/models/model_20250704_140031_checkpoint.pt
Validating:   0%|                                                                                                                 | 0/1939 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)

=== Example ===
Reference: A dark-skinned woman, wearing a pink top and brown and white skirt, is crossing the street.
Generated: A woman in a red dress is holding a child.

=== Example ===
Reference: Two soldiers are going into a tent with men and women watching them.
Generated: A group of people are standing in front of a tent.

=== Example ===
Reference: A young girl in a blue shirt is bending over playing with a pile of sand on what seems to be a beach.
Generated: A girl in a pink shirt is playing in the sand.

=== Example ===
Reference: There is a guy wearing stilts playing the trombone in front of a mural.
Generated: A man is standing on a beach with a bandanna over his head.

=== Example ===
Reference: A man wearing a black sweater cooks food in a pan while standing in a cluttered kitchen.
Generated: A man in a black shirt is cooking food.
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 1939/1939 [20:37<00:00,  1.57it/s]

Validation loss: 1.3086 | Average BLEU: 0.0676





Train Loss: 1.2257
Model saved to /root/models/model_20250704_140246_checkpoint.pt
Validating:   0%|                                                                                                                 | 0/1939 [00:00<?, ?it/s]
=== Example ===
Reference: A dark-skinned woman, wearing a pink top and brown and white skirt, is crossing the street.
Generated: A woman in a red dress is walking down the street with a child in a white dress.

=== Example ===
Reference: Two soldiers are going into a tent with men and women watching them.
Generated: A group of people are standing in a tent.

=== Example ===
Reference: A young girl in a blue shirt is bending over playing with a pile of sand on what seems to be a beach.
Generated: A little girl is digging in the sand.

=== Example ===
Reference: There is a guy wearing stilts playing the trombone in front of a mural.
Generated: A man in a black shirt is playing a guitar on a bridge.

=== Example ===
Reference: A man wearing a black sweater cooks food in a pan while standing in a cluttered kitchen.
Generated: A man in a black shirt is cooking.
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 1939/1939 [29:30<00:00,  1.10it/s]

Validation loss: 1.2525 | Average BLEU: 0.0727

Epoch 4/5


















[Step 3550] Running Avg Loss: 1.1752
Reference: The dog is walking in the snow.
Generated: A brown dog is running through the snow.

[Step 3600] Running Avg Loss: 1.2059
Reference: Young women chat after a stroll along the beach line.
Generated: A group of people are standing on the beach.

[Step 3650] Running Avg Loss: 1.1684
Reference: Fishermen casting a large net out to sea.
Generated: A group of people are working on a beach.

[Step 3700] Running Avg Loss: 1.1963
Reference: Men race their bikes on a road.
Generated: Three men are riding bicycles on a road.

[Step 3750] Running Avg Loss: 1.1682
Reference: A child wearing a jacket that says USA is shoveling snow.
Generated: A child in a red jacket and black pants is shoveling snow from a sidewalk.

[Step 3800] Running Avg Loss: nan
Reference: Young girl at a festival wearing a hat with cow horns and sunglasses pulling a milk crate on wheels.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 3850] Running Avg Loss: nan
Reference: A boy wearing blue shorts is bouncing a basketball in front of the net.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 3900] Running Avg Loss: nan
Reference: A girl with black hair in athletic clothing tossing a football.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 3950] Running Avg Loss: nan
Reference: A group of people are running through a street.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 4000] Running Avg Loss: nan
Reference: A couples is walking outside under some trees while the flower petals are falling.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 4050] Running Avg Loss: nan
Reference: Girl making a painting on the wall with spray.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 4100] Running Avg Loss: nan
Reference: A man in a green shirt playing an acoustic guitar.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 4150] Running Avg Loss: nan
Reference: Several children practicing martial arts with wooden swords
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 4200] Running Avg Loss: nan
Reference: Two males are sitting in back of a pick up truck.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 4250] Running Avg Loss: nan
Reference: Two men are working on a upside down purple and pink girls bicycle, while two little girls, a boy, and an older woman watch.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 4300] Running Avg Loss: nan
Reference: Man with a shoulder bag standing on concrete.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 4350] Running Avg Loss: nan
Reference: Native Americans are dressed up in native clothing and are participating in an activity together.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 4400] Running Avg Loss: nan
Reference: A group of men in reflective gear are holding light sticks while standing on a wooden floor that has outdoor lighting.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 4450] Running Avg Loss: nan
Reference: Person wearing black and tan winter jacket sitting on log by waters edge.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 4500] Running Avg Loss: nan
Reference: A girl wearing colored striped pants brushing her teeth next to a white couch.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 4550] Running Avg Loss: nan
Reference: A woman spikes a volleyball as the opposing player tries to block the ball.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 4600] Running Avg Loss: nan
Reference: A boy is raising his hand to answer a question.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 4650] Running Avg Loss: nan
Reference: Person walking through the snow leafless trees in background
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 4700] Running Avg Loss: nan
Reference: A red car is caught in an explosion.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 4750] Running Avg Loss: nan
Reference: The young girl cools off by sliding on the water slide.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 4800] Running Avg Loss: nan
Reference: A man wearing a cap juggles a tennis racket, a bowling ball, and a basketball.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 4850] Running Avg Loss: nan
Reference: Young adults posing for a photo at night, somewhat chilly outside.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 4900] Running Avg Loss: nan
Reference: A woman cuts a cake while the woman next to her holds out a plate.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 4950] Running Avg Loss: nan
Reference: A baseball player kicks up dirt sliding in front of a catcher.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 5000] Running Avg Loss: nan
Reference: An outdoor scene of people either sitting down or standing up with a little child walking towards the pigeons.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 5050] Running Avg Loss: nan
Reference: A man in a light blue long-sleeved shirt wearing a red and white scarf on his head stands beside a makeshift fruit stand.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 5100] Running Avg Loss: nan
Reference: Dog barking
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 5150] Running Avg Loss: nan
Reference: A white-shirted man is leaping from the stage into the crowd.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 5200] Running Avg Loss: nan
Reference: Punk rock teenager walks with a mean look on his face.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 5250] Running Avg Loss: nan
Reference: A guy in a blue shirt next to a man in a white shirt who is looking at a guy in an orange shirt.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 5300] Running Avg Loss: nan
Reference: A man in a blue jumpsuit and a brown hat wearing a black tie, boots and goggles sits with his legs crossed in the street.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 5350] Running Avg Loss: nan
Reference: A line of people hold candles and signs to promote saving trees.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 5400] Running Avg Loss: nan
Reference: A waitress is serving customers at a restaurant.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 5450] Running Avg Loss: nan
Reference: A woman in a green dress stops to look at her phone.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 5500] Running Avg Loss: nan
Reference: A man fishing on some large rocks while the sun is low in the sky.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 5550] Running Avg Loss: nan
Reference: People having fun at a party.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 5600] Running Avg Loss: nan
Reference: Eight people wearing red suits and carrying red bags are walking across a bridge.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 5650] Running Avg Loss: nan
Reference: A woman wearing a sweatshirt and jeans is at a laundry mat is putting ERA in the washing machine.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 5700] Running Avg Loss: nan
Reference: A man in a black and blue wetsuit is out on the ocean sailing on his surfboard.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 5750] Running Avg Loss: nan
Reference: A group of greyhound dogs racing with muzzles covering their noses.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 5800] Running Avg Loss: nan
Reference: Cowboy with a purple shirt and black vest trying to hold on during a wild horse ride.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 5850] Running Avg Loss: nan
Reference: Blond male sculptor working on a abstract art sculpture in studio.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 5900] Running Avg Loss: nan
Reference: A man wearing a yellow t-shirt is shouting into a microphone while an audience is listening.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 5950] Running Avg Loss: nan
Reference: A man sits in front of a wall with art pictures on it.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 6000] Running Avg Loss: nan
Reference: A boy in red baseball gear makes a pitch, in front of a trailer park.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 6050] Running Avg Loss: nan
Reference: An older man sites on a cushioned bench with his hands crossed in his lap.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 6100] Running Avg Loss: nan
Reference: A guy working under an umbrella on a sewing project.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 6150] Running Avg Loss: nan
Reference: Several men having a drink at a bar.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 6200] Running Avg Loss: nan
Reference: A group of people gathers on the grass in a backyard with tents, tables, and chairs set up.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 6250] Running Avg Loss: nan
Reference: A woman selling her homemade products.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 6300] Running Avg Loss: nan
Reference: A man in a white shirt and dark pants jumping to an extreme height on a warm summer day.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 6350] Running Avg Loss: nan
Reference: A person is standing by some stairs in colorful clothes and goggles.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 6400] Running Avg Loss: nan
Reference: A man and woman are having a conversation on a park bench.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 6450] Running Avg Loss: nan
Reference: A man wearing athletic biking clothing and a helmet is riding his bike down a ramp made of wooden boards.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 6500] Running Avg Loss: nan
Reference: A man picks up a dripping wet woman happily.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 6550] Running Avg Loss: nan
Reference: A man in a hat looks at a waterfall.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[Step 6600] Running Avg Loss: nan
Reference: A child looking through a telescope on a playground.
Generated: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```


nohup torchrun --nproc-per-node=1
npin_memory
nworker=16

Env var TOKENIZER_PARALLELISM

Use prefix for instruction
e.g. "A photo of"

Show attention maps

BLEU, ROUGE, METEOR etc



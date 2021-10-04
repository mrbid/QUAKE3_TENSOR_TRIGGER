# QUAKE3_TENSOR_TRIGGER
A series of machine learning trigger bots for Quake3 Arena &amp; Quake Live.

Where do I start... Well... This all started around Feburary of 2020 with https://github.com/quake3aimbot it was my first attempt at working with perceptrons and it lead to me creating [this article](https://medium.com/swlh/training-a-neural-network-to-autoshoot-in-fps-games-e105f27ec1a0) which describes the process of how I made my first simple CNN like neural network for automatic tigger in Quake3.

After that I got a little deeper into [Feedforward Neural Networks](https://en.wikipedia.org/wiki/Feedforward_neural_network) and started the [TFCNN project](https://github.com/TFCNN) which is somewhat ongoing.

I then decided I wanted to make an auto trigger bot for CS:GO using TFCNN and started [a whole 6 series of articles](https://james-william-fletcher.medium.com/list/fps-machine-learning-autoshoot-bot-for-csgo-100153576e93) based on this where I started trying to apply TFCNN and failed, [then I wrote a CNN in C](https://github.com/TFCNN/Projects/blob/main/TBVGG3_NAG.h) based on [VGG network architecture](https://www.robots.ox.ac.uk/~vgg/research/very_deep/) and had some really great success, although with a lot of blood & sweat [so to speak](https://www.google.com/speech-api/v1/synthesize?text=so%20to%20speak&enc=mpeg&lang=en-gb&speed=0.4&client=lr-language-tts&use_google_only_voices=1).

Wanting to take the project a little further but also knowing that writing all the code in C from scratch was serverly limiting my ability to rapidly prototype I took up [Tensorflow Keras](https://keras.io/about/) using Python and really honed in on that success.

I then got bored of CG:GO, and focused my efforts back on Quake3 a game that I preferred to spend my time playing anyway.. and it was open source. It's one of those games that is unlikely to ever cease to exist for this very reason.

I set my sights back on the aqua blue bones model which actually turned out to be a lot harder to target than a player head in CS:GO.

Here we are now, this repository is the fruits of my labour upto this point.

## Prerequisites
`sudo apt install clang gcc libxdo-dev libxdo3 libespeak1 libespeak-dev espeak xterm`

## How To
- With the SimpleCNN models you can just go and [grab the bin directly here](https://github.com/mrbid/QUAKE3_TENSOR_TRIGGER/tree/main/Binaries) or just compile from source by invoking the `compile.sh` file in the local directory.
- With the Keras models you will first want to invoke the `train.py` file in the respective `/Trainer` directory. This will train the model, also compile the C client which interfaces with the Python daemon. Then you can go into the `/PredictBot` directory and run `exec.sh` which will launch two console windows, the C program in Xterm and the Python daemon in the original console window exected by the exec script.
- You're going to need to setup Tensorflow and Python3, a guide on how to do this is included in [this article](https://james-william-fletcher.medium.com/creating-a-machine-learning-auto-shoot-bot-for-cs-go-part-6-af9589941ef3).
- You should be good to go.

## Wisdom
When it comes to a CNN or an FNN it's this simple.
- The CNN is much more accurate at what its trained to detect, but lacks the locality an FNN has, although this is a doubled edged sword...
- The FNN is much more generalised, although I allow users to specify a custom amount of Units this is simpily for novelty, the only useful method of using an FNN in computer vision for this purpose is to have one weight per pixel colour channel and then feed that all into one unit which outputs a binary decision (0 = not a target, 1 = a target) what this does is essentially creates an averaged cookie-cutter filter/map, which is what causes the larger range of generalisation. In some respects this can be a good thing, but it does come with a slightly higher rate of missfire. This can be mitigated by increasing the `REPEAT_ACTIVATION` definition in the `quakelive_bluebones_autoshoot.c` file to some degree.

A CNN will train and execute faster on a GPU as where a FNN will train and execute faster on a CPU because an FNN is just a [FMA operation](https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation) which CPU's have [AVX](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions) for.
Generally you don't want to multi-thread these solutions much because you don't want to eat into the games processing resources too much, but if you do have some excess CPU cores to spare the qay I recommend scaling up the FNN solution is to train one model per Quake3 player model, then execute each trained model at the same time across multiple CPU cores. This way your VRAM to RAM operation which is generally limited to ~60 FPS can grab a frame and the CPU can process multiple trained networks on the frame at the same time _(one per Quake3 player model)_ and return a attack or not to attack signal faster than bobs your uncle.

The SimpleCNN models are still the best as they only use a 3x3 sample window so they are much more accurate, and, they have much less missfire. Crazy huh, so much work to make these 28x28 models which are higher in computational cost, time invested, etc, and yet still less effective.

The next best model, or what I consider the forefront of the evolution at the moment is the 28x28 `QUAKE3_CNN` project. The 3x3 bot _(SimpleCNN)_ is a done deal, but the CNN has much room for improvement. The FNN on the other hand is like a nice side-toy, a biproduct of the CNN.

## Oddities
`QUAKE3_CNN` only has one C client as where `QUAKE3_FNN` has two, one in the root directory which compiles a standalone version in C which uses AVX FMA extensions and one in the `/PredictBot` directory which interfaces with the Python daemon.

## What now?
Well it depends what your intentions are, if you're just after pwning some noobz, then join this excessiveplus server [173.199.79.107:27974](http://elitez.eu) and go wild, this is only really decent on Instagib game modes. Or join a QuakeLive Instagib server, and while you're at it tell everyone to download ioQuake3 because it's better and more people play it.

But I prefer you didn't cheat unless it's on that specific server I linked because _everyone_ cheats on it, or QuakeLive because obviously I am not a great fan of QuakeLive.

What I would prefer is if you could contribute to the projects dataset, by submitting a zipped folder of all your captured samples somehow via an issue on this repo. But please, try to sort the samples first, just because the bot says all the samples in the targets directory are targets is not always the case, the general rule of thumb is:
- If the target is not dead center of the 28x28 window, it's a "nontarget" and if it is dead center, then it's a "target".

## Conclusion

Just trying to make the best computer vision model & dataset for Quake3.





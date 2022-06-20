![](img/banner.svg)
# PenPal
An end to end conversational agent via physical handwriting.

We use Google Cloud Vision to read handwritten text from the page, GPT-3 from OpenAI to complete the text, neural network handwriting synthesis to synthesize the stroke trajectory for natural looking writing, and then finally the AxiDraw SDK to write the response on the paper. An Arduino Nano with a photoresistor in a custom 3D printed pen holder controls the "handoff" between human and machine.

We make use of the handwriting synthesis experiments in the paper [Generating Sequences with Recurrent Neural Networks](https://arxiv.org/abs/1308.0850) by Alex Graves, ported to Tensorflow by [Sean Vasquez](https://github.com/sjvasquez/handwriting-synthesis), and set up to control an [AxiDraw v3 pen plotter](https://wiki.evilmadscientist.com/AxiDraw).
![](img/banner.svg)
# PenPal
An end to end conversational agent via physical handwriting.

We use Google Cloud Vision to read handwritten text from the page, GPT-3 from OpenAI to complete the text, neural network handwriting synthesis to synthesize the stroke trajectory for natural looking writing, and then finally the AxiDraw SDK to write the response on the paper. An Arduino Nano with a photoresistor in a custom 3D printed pen holder controls the "handoff" between human and machine.

We make use of the handwriting synthesis experiments in the paper [Generating Sequences with Recurrent Neural Networks](https://arxiv.org/abs/1308.0850) by Alex Graves, ported to Tensorflow by [Sean Vasquez](https://github.com/sjvasquez/handwriting-synthesis), and set up to control an [AxiDraw v3 pen plotter](https://wiki.evilmadscientist.com/AxiDraw).

## Installation
Install all the requirements in `requirements.txt`, of particular note are: `tensorflow`, `opencv`, `pyfirmata`, `google-cloud-vision`, `openai`, and the [AxiDraw SDK](https://axidraw.com/doc/py_api/#installation)

Set environment variables: `GOOGLE_APPLICATION_CREDENTIALS` and `OPENAI_API_KEY`

Confirm that your webcam is connected to port 0, your Arduino is connected to `tty/usbserial_0001`, your Google Cloud Vision account is configured for billing, and your OpenAI API account is configured appropriately.

## Usage
The codebase is a bit difficult to navigate right now but most of the functionality is in `demo.py`.

First, confirm that your AxiDraw is oriented at the top of your page, the pen carriage is set to the top left, the pen is loaded into the pen fixture at the appropriate height, the webcam is pointed at the page, and the human pen is in the pen holder.

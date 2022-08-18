# PenPal
An IRL language model playground via pen & paper.

![IMG_6953](https://user-images.githubusercontent.com/578640/185456872-fdfc5fdf-4053-4630-9a47-f570e6fc2f06.jpg)

We use Google Cloud Vision to read handwritten text from the page, GPT-3 from OpenAI to complete the text, neural network handwriting synthesis to synthesize the stroke trajectory for natural looking writing, and then finally the AxiDraw SDK to write the response on the paper. An Arduino Nano with a photoresistor in a custom 3D printed pen holder controls the "handoff" between human and machine.


## Installation
1. Install all the requirements in `requirements.txt`. 
Of particular note are: `tensorflow`, `opencv`, `pyfirmata`, `google-cloud-vision`, `openai`, and the [AxiDraw SDK](https://axidraw.com/doc/py_api/#installation).

2. I would recommend using a virtual environment, ideally in Anaconda so that you aren't wasting time building wheels.
[This guide to installing Tensorflow on Mac M1 saved my life](https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706)

3. Set environment variables: `GOOGLE_APPLICATION_CREDENTIALS` and `OPENAI_API_KEY`

4. Confirm that your webcam is connected to port 0, your Arduino is connected to `/dev/cu.usbserial-0001`, your Google Cloud Vision account is configured for billing, and your OpenAI API account is configured appropriately.

### Configuration (physical)
* [x] 8.5" x 5.5" paper
* [x] AxiDraw oriented at the top of your page
* [x] Pen carriage set to the top left
* [x] Pen loaded into the pen fixture at the appropriate height
* [x] Webcam pointed at the page from above and the top of the paper (as in photo)
* [x] Human pen in the pen holder.

## Usage
The codebase is a bit difficult to navigate right now but most of the functionality is in `demo.py`.
1. Confirm that the physical configuration is set.
2. Run `python demo.py`
3. Remove the pen from the holder, write something on the page, place the page back in position, and return the pen to the holder
4. Wait for the plotter to finish writing before writing again.

There are a lot of options for configuring cnn-vis; this page will show some examples to get you started.

When making Inceptionism-style images, one of the most important parameters is the target layer. Using lower layers will tend to emphasize low-level image features, while higher layers will tend to emphasize object parts. As an example, we will start with Edvard Munch's classic painting ["The Scream"](https://en.wikipedia.org/wiki/The_Scream) and modify it using several different CNN layers:

<img src="http://cs.stanford.edu/people/jcjohns/cnn-vis-examples/initial-imgs/scream.jpg" width="400px">
<img src="http://cs.stanford.edu/people/jcjohns/cnn-vis-examples/example7.png" width="400px">
<img src="http://cs.stanford.edu/people/jcjohns/cnn-vis-examples/example8.png" width="400px">
<img src="http://cs.stanford.edu/people/jcjohns/cnn-vis-examples/example9.png" width="400px">

* **Upper left**: The original image.
* **Upper right**: Produced using the script [example7.sh](examples/example7.sh) which amplifies the relatively early `inception_3a/1x1` layer.
* **Lower left**: Produced using the script [example8.sh](examples/example8.sh) which amplifies the relatively early `inception_3a/3x3_reduce` layer.
* **Lower right**: Produced using the script [example9.sh](examples/example9.sh) which amplifies the later layer `inception_4d/output` layer, which is a bit later. Amplifying this layer causes animal parts to appear.

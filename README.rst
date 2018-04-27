*****
aca-tensorflow
*****
.. image:: https://travis-ci.org/GuidoBallabio/aca-tensorflow.svg?branch=master
    :target: https://travis-ci.org/GuidoBallabio/aca-tensorflow

ACA project on implementation, evaluation and comparison of CNNs:

Train a CNN model for Cifar-10 or MNIST dataset with Tensorflow then use Quantization methods present in Tensorflow framework to generate models with possibly different quantization/compression methdos. Compare all the models on the basis of model accuracy in inference forwardpass, execution time performance, size of the models.

Details:
########

Important Notes:
****************

Try to make timing measurements as accurate as possible, meaning:
Do not run in the virtual machine unless you are using some kind of simulator
Limit the number of background applications (close browsers, video-playback, ....)
If you are using C/C++, please avoid using malloc-free (or new-delete) inside measurements as much as possible
Similarly, do not open-close and/or read-write from files inside measurements
In the case there is too much noise in your system, please measure multiple times and report statistics rather than absolutes. 
To reduce the effects of caches, you should run N times your application successively and discard the first 5-10 measurements (This is sometimes called cache-warming).

A good tutorial for Neural Network, if you have no idea what they are. They are not strictly necessary for the projects, but it may give better perspective and context to you on ”why you are doing X project?”.

* http://cs231n.stanford.edu/


----------------------------------------------

Name:	Tensorflow Neural Network Quantization
Code:	P17

Type:		Programming(Python)
Max Points:	12 (6+6)

Description:
For real world application, convolutional neural network(CNN) model can take more than 100MB of space and can be computationally too expensive. Therefore, there are multiple methods to reduce this complexity in the state of art. The goal of this project is to apply some neural network quantization techniques with high-level frameworks like Tensorflow and observe the effects of quantization both on the accuracy of the network and the execution performance of the neural network during inference phase. In the project we are not interested in the training phase performance. The project requires that two or more models trained for Cifar10 or MNIST dataset with Tensorflow and with possibly different quantization methodologies. 
The comparison of the models will be based on the execution time, model size and cache utilization of the inference run of the neural networks that are trained. The effectiveness of comparison between different networks are essential for this project therefore it is strongly suggested for students to train networks with diverse characteristics. The inference run might be tested on CPU platforms and the cache utilization can be gathered from Linux Perf or Cachegrind tools.

References (Starting Points):

* https://www.tensorflow.org/performance/quantization
* http://valgrind.org/docs/manual/cg-manual.html
* https://perf.wiki.kernel.org/index.php/Main_Page

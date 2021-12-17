# zenoh-iftdt-demo


:warning: This example works only on Linux and it require OpenCV to be installed, please follow the instruction on the [OpenCV documentation](https://docs.opencv.org/4.5.2/d7/d9f/tutorial_linux_install.html) to install it.

:warning: You need a machine equipped of a webcam in order to run this example.

:warning: This example works only on Linux and it require OpenCV with CUDA enabled to be installed, please follow the instruction on [this gits](https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7) to install it.

:warning: This example works only on Linux and it require a **CUDA** capable **NVIDIA GPU**, as well as NVIDIA CUDA and CuDNN to be installed, please follow [CUDA instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and [CuDNN instructions](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html).

:warning: You need a machine equipped of a webcam in order to run this example.

:warning: You need to download a YOLOv3 configuration, weights and classes, you can use the ones from [this GitHub repository](https://github.com/sthanhng/yoloface).

Zenoh Flow + Zenoh-pico Home Automation Demo


# Example Application

This applications leverages the Zenoh famility (Zenoh, Zenoh-pico, Zenoh Flow ), to implement a simple smart-home application.
Application components runs locally, in particular all the AI and DNN are running locally on a CUDA powered machine.
All pictures never leave the local network, the connecivity to the cloud is only used to send commands the smart lamps.

The graph composing the application is described in `dataflow.yml` and illustrated for convenience in the picture below.

![Application Graph](application-graph.png)

From left-to-right, we can see that this applications has two sources, a camera, and a luminosity sensor.
Both of them produces data at different pace.

The camera produces frame and a pace defined in the `dataflow.yml` (*15fps*) and send those frames to the Face-detection components, that uses YOLO to cut faces and send those faces to the Face-recognition, that then sends out an array of people.

On the other arm, the sensor sends raw data to an operator that normalizes and then merges with the upper arm in the Sensor Fusion.

This one based on the array of people and the luminosity in the room decides if and which lamp to turn on (or off).

Then based on this computation a dictionary of `{'lamp_name':'luminosity_for_lamp'}` is sent to the sink that interacts with the cloud provider to turn on (or off the light).

As already said in the beginning, all the components of the applications runs locally, no cloud services are involved of the AI/DNN nor for the sensor normalization and processing.

The only cloud service is used when sending the commands to the smart lamps.

# How to run

0. Get the YOLO config from [this GitHub](https://github.com/sthanhng/yoloface) and place it inside the `face-detect` folder

1. Train the face recognition using: [this GitHub](https://github.com/gabrik/face-recog-tf2) and then place the `encodings.pkl` inside the `recognition` folder.

2. Clone and build the zenoh flow runtime example

```bash
git clone https://github.com/atolab/zenoh-flow-examples
cd zenoh-flow-examples
cargo build --release --bin runtime
```

3. Clone, build and install zenoh-flow-python bindings

```
git clone https://github.com/atolab/zenoh-flow-python
cd zenoh-flow-python/zenoh-flow-python
pip3 install -r requirements-dev.txt
python3 setup.py bdist_wheel
pip3 install dist/*.whl
cd ..
cargo build --release --all-targets
```

4. Update the `dataflow.yml` file according to your configuration (APIs keys, tokens....)

5. Run the demo: `../zenoh-flow-examples/target/release/runtime -r debug -g dataflow.yml -l loader-config.yml`


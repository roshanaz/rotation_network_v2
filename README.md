# Rotation Angle Prediction

Model that predicts the rotation angle between two images.

## How to run it


### Prerequisites

- Python 3.10 or higher
- uv package manager (<code>pip install uv</code> or see [here](https://docs.astral.sh/uv/getting-started/installation/) for more info)

### Installation (Mac) 

```bash
cd rotation_network_v2
# install 
uv python install 3.10
uv run --group mac  -p 3.10 inference.py test_images/dog4.png test_images/dog4_rotated.png
```
**_NOTE:_** It takes roughly 6 seconds to load the model and run the inference on CPU. 

### Installation (Linux) 

```bash
cd rotation_network_v2
# install 
uv python install 3.10
uv run --group linux  -p 3.10 inference.py test_images/dog4.png test_images/dog4_rotated.png
```

### Docker

You can also run it using Docker. 

Following is an example of running the docker image `roshie/rotation` for the image pair (dog4.png and dog4_rotated.png) in the `test_images` folder, you can copy your image pairs inside the `test_images` folder.

```bash
cd rotation_network_v2

docker run -v $PWD/test_images:/imgs roshie/rotation /imgs/dog4.png /imgs/dog4_rotated.png
```

=======
# rotation_network_v2
includes train and inference code

# CLEVR Image Generation

This repository modifies the **CLEVR dataset generation code** from [Meta Research](https://github.com/facebookresearch/clevr-dataset-gen) (BSD License) to generate images with a specific number of objects.

## Setup
Install [Blender](https://www.blender.org/). The code is adapted for Blender 2.93, but other versions may also work.

Blender uses its own Python distribution (e.g., Python 3.9). You need to add the `clevr_generation` directory to Blender's Python path:

```
echo $PWD/clevr_generation >> $BLENDER/$VERSION/python/lib/$PYTHON_VERSION/site-packages/clevr.pth
```
where $BLENDER is the directory where you installed Blender, $VERSION is its version (e.g., 2.93), and $PYTHON_VERSION is the Python version that came with your installation of Blender (e.g., python3.9).

For example:

```
echo $PWD/clevr_generation >> ../blender/2.93/python/lib/python3.9/site-packages/clevr.pth
```

## Image generation

Run:

```
$BLENDER/blender --background --python render_images.py -- [args]
```

If rendering on a cluster without audio drivers, add `-noaudio`:

```
$BLENDER/blender --background -noaudio --python render_images.py -- [args]
```

### Example: Generating 3-object scenes using GPU
```
cd clevr_generation
my_path_to_blender/blender --background -noaudio --python render_images.py -- --use_gpu 1 --split 3obj --num_objects 3
```

See the [CLEVR image generation guide](https://github.com/facebookresearch/clevr-dataset-gen/tree/main/image_generation) for settings details, or run:

```
python render_images.py --help
```
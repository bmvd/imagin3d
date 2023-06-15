import os
import sys
import torch
import subprocess as sp
from tkinter import filedialog
import replicate



##!nvidia-smi
##! pip install plotly -q
##!git clone https://github.com/openai/point-e
##! pip install -e .

#Specify Replicate API Token
os.environ['REPLICATE_API_TOKEN'] = 'InsertReplicateTokenHere'

#Select Image From Folder
filepath = r"C:\Users\specifiedPath"
#filepath = filedialog.askopenfilename(initialdir=r"C:\Users\20212381\AI_Project\Images")
filepath = filedialog.askopenfilename(initialdir=r"C:\Users\specifiedPath")

##print(filepath)


print('Importing Image')
image = open(filepath, "rb")

print('Fetching prompt')
output = replicate.run(
    "methexis-inc/img2prompt:50adaf2d3ad20a6f911a8a9e3ccf777b263b8596fbd2c8fc26e8888f8a0edbb5",
    input={"image": image}
)
##print(output)

#Start Cleaning Prompt
splitChar = ','
prompt = output.split(splitChar, 1)[0]

prompt = prompt.replace('a drawing of a ', '')
prompt = prompt.replace('a drawing of ', '')
prompt = prompt.replace(' on a piece of paper', '')
prompt = prompt.replace('a pencil drawing of a ', '')
prompt = prompt.replace('a pencil drawing of', '')
prompt = prompt.replace('a black and white drawing of a', '')
prompt = prompt.replace('a black and white drawing of ', '')
prompt = prompt.replace('a close up of', '')

##print("Prompt:", prompt)

print("Starting Model")

#Changing Directory
path = os.path.abspath("point-e")
sys.path.append(path)

"""### Specific Imports"""

from tqdm.auto import tqdm

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

"""### Models"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(' creating base model...')
base_name = 'base40M-textvec'
base_model = model_from_config(MODEL_CONFIGS[base_name], device)
base_model.eval()
base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

print(' creating upsample model...')
upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
upsampler_model.eval()
upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

print(' downloading base checkpoint...')
base_model.load_state_dict(load_checkpoint(base_name, device))

print(' downloading upsampler checkpoint...')
upsampler_model.load_state_dict(load_checkpoint('upsample', device))

sampler = PointCloudSampler(
    device=device,
    models=[base_model, upsampler_model],
    diffusions=[base_diffusion, upsampler_diffusion],
    num_points=[1024, 4096 - 1024],
    aux_channels=['R', 'G', 'B'],
    guidance_scale=[3.0, 0.0],
    model_kwargs_key_filter=('texts', ''), # Do not condition the upsampler at all
)

# Set a prompt to condition on.
promptUsed = prompt

# Produce a sample from the model.
samples = None
for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[promptUsed]))):
    samples = x

pc = sampler.output_to_point_clouds(samples)[0]

fig = plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))

import plotly.graph_objects as go

fig_plotly = go.Figure(
        data=[
            go.Scatter3d(
                x=pc.coords[:,0], y=pc.coords[:,1], z=pc.coords[:,2],
                mode='markers',
                marker=dict(
                  size=2,
                  color=['rgb({},{},{})'.format(r,g,b) for r,g,b in zip(pc.channels["R"], pc.channels["G"], pc.channels["B"])],
              )
            )
        ],
        layout=dict(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            )
         ),
     )

fig_plotly.show(renderer="colab")

from point_e.util.pc_to_mesh import marching_cubes_mesh

device = torch.device('cuda' if torch.cuda.is_available()else 'cpu')

print(' creating SDF model...')
name = 'sdf'
model = model_from_config(MODEL_CONFIGS[name], device)
model.eval()

print(' loading SDF model...')
model.load_state_dict(load_checkpoint(name,device))

import skimage.measure as measure

mesh = marching_cubes_mesh(
    pc=pc,
    model = model,
    batch_size = 4096,
    grid_size = 32,
    progress = True,
)
print("Saving File")
filepath = filepath.replace('.jpg', '.ply')
filepath = filepath.replace('originalPictureLocation', '/AI_Project/Models')

fileName = filepath.replace("FileLocation", "")
fileName = fileName.replace(".ply", "")

promptname = prompt.replace(' ', '_')
promptname = promptname.replace('\n', '')
print(filepath)
print(promptname)
filepath = filepath.replace(fileName, promptname)
print(filepath)
with open(filepath, 'wb') as f:
  mesh.write_ply(f)

print("PLY Created")
print("Opening PLY)")

os.startfile(filepath)
sys.exit()
# DrugFlow & FlexFlow

<a href="https://openreview.net/forum?id=g3VCIM94ke"><img src="https://img.shields.io/badge/ICLR-2025-brown.svg" height=22.5></a>

Code repository for "Multi-domain Distribution Learning for De Novo Drug Design" by Arne Schneuing*, Ilia Igashov*, Adrian W. Dobbelstein, Thomas Castiglione, Michael M. Bronstein, and Bruno Correia 

![](docs/drugflow.jpg)

## Abstract
We introduce DrugFlow, a generative model for structure-based drug design that integrates continuous flow matching with discrete Markov bridges, demonstrating state-of-the-art performance in learning chemical, geometric, and physical aspects of three-dimensional protein-ligand data. We endow DrugFlow with an uncertainty estimate that is able to detect out-of-distribution samples. To further enhance the sampling process towards distribution regions with desirable metric values, we propose a joint preference alignment scheme applicable to both flow matching and Markov bridge frameworks. Furthermore, we extend our model to also explore the conformational landscape of the protein by jointly sampling side chain angles and molecules.

## Setup

### Conda Environment

Create a conda/mamba environment 
```bash
conda env create -f environment.yaml -n drugflow
conda activate drugflow
```

and add the Gnina executable for docking score computation
```bash
wget https://github.com/gnina/gnina/releases/download/v1.1/gnina -O $CONDA_PREFIX/bin/gnina
chmod +x $CONDA_PREFIX/bin/gnina
```
### Conda Environment for Mac (Apple Silicon)

Create a conda/mamba environment 
```
conda env create -f environment_Mac.yaml -n drugflow
conda activate drugflow
```

Gnina is not available for Apple Silicon as it depends on libmolgrid, which heavily depends on cuda kernels. Qvina is installed in the conda environment as an alternative for docking scoring.


### Docker Container

A pre-built Docker container is available on [DockerHub](https://hub.docker.com/r/igashov/drugflow):

```bash
docker pull igashov/drugflow:0.0.3
```

## Basic Usage

To sample molecules for a protein target:

```bash
# Download a model
wget -P checkpoints/ https://zenodo.org/records/14919171/files/drugflow.ckpt

# Generate molecules
python src/generate.py \
  --protein examples/kras.pdb \
  --ref_ligand examples/kras_ref_ligand.sdf \
  --checkpoint checkpoints/drugflow.ckpt \
  --output examples/samples.sdf
```

For more options, see
```bash
python src/generate.py --help
```

## Models

Please find model checkpoints [here](https://zenodo.org/records/14919171) or use the download links below:
- [DrugFlow](https://zenodo.org/records/14919171/files/drugflow.ckpt?download=1)
- [DrugFlow + confidence head](https://zenodo.org/records/14919171/files/drugflow_ood.ckpt?download=1)
- [FlexFlow](https://zenodo.org/records/14919171/files/flexflow.ckpt?download=1)
- [DrugFlow after preference alignment](https://zenodo.org/records/14919171/files/drugflow_pa_comb.ckpt?download=1)


## Dataset preparation

### Pre-processed dataset
The preprocessed dataset is available on Zenodo
```bash
wget https://zenodo.org/records/14919171/files/processed_crossdocked.zip
unzip processed_crossdocked.zip
```

### (Optional) running pre-processing locally

To process the raw dataset locally, first download and extract the CrossDocked dataset as described by the authors of Pocket2Mol: https://github.com/pengxingang/Pocket2Mol/tree/main/data.

Specify input and output directories
```bash
CROSSDOCKED_DATA=...  # location at which the dataset was extracted
PROCESSED_DATA=...  # location at which the processed dataset will be stored
```

Then, preprocess the data for DrugFlow
```bash
python src/data/process_crossdocked.py $CROSSDOCKED_DATA \
       --outdir $PROCESSED_DATA \
       --flex
```

### (Optional) Pre-processing of a custom preference alignment dataset

To create a dataset for preference alignment (PA), first, download the preprocessed [dataset](#pre-processed-dataset).

Then, [sample](#sampling-for-all-proteins-in-the-test-set) a synthetic dataset using a pretrained reference model and evaluate the samples by first specifying input and output directories and [evaluate](#evaluating-samples) the samples.

```bash
PREPROCESSED_DATA=...  # Location of the preprocessed data directory
SAMPLES_DIR=...  # Location where the sampled dataset is stored
EVALUATED_DATA=...  # Directory for evaluation output
```

Specify input and output directories for the PA dataset:

```bash
PROCESSED_DATA=...  # Location at which the processed dataset will be stored
METRICS_PATH=$EVALUATED_DATA/metrics_detailed.csv
CRITERION=...  # Preference alignment criterion ('reos.all', 'medchem.sa', 'medchem.qed', 'gnina.vina_efficiency', or 'combined')
```
Finally, preprocess the data for DrugFlow-PA:

```bash
python src/data/process_dpo_dataset.py \
       --smplsdir $SAMPLES_DIR \
       --basedir $PROCESSED_DATA \
       --datadir $PREPROCESSED_DATA \
       --dpo-criterion $CRITERION \
       --metrics-detailed $METRICS_PATH \
       --ignore-missing-scores
```

## Training

Example config files are provided for:
- DrugFlow: `CONFIG=configs/training/drugflow.yml`
- FlexFlow: `CONFIG=configs/training/flexflow.yml`
- Preference alignment: `CONFIG=configs/training/preference_alignment.yml`

Create a symlink to the processed dataset and for the output directory
```bash
LOGDIR=...  # where checkpoints, and validation outputs will be saved
ln -s $PROCESSED_DATA processed_crossdocked
ln -s $LOGDIR runs
```
Alternatively, you can change the corresponding paths in the config files.

To launch the training job for the DrugFlow base model, for example, run
```bash
python src/train.py --config $CONFIG
```


## Inference

### Checkpoints 

Pretrained checkpoints can be downloaded from Zenodo with

```bash
# Base DrugFlow model
wget -P checkpoints/ https://zenodo.org/records/14919171/files/drugflow.ckpt

# DrugFlow + confidence head
wget -P checkpoints/ https://zenodo.org/records/14919171/files/drugflow_ood.ckpt

# FlexFlow
wget -P checkpoints/ https://zenodo.org/records/14919171/files/flexflow.ckpt

# DrugFlow after preference alignment
wget -P checkpoints/ https://zenodo.org/records/14919171/files/drugflow_pa_comb.ckpt
```

### Sampling for all proteins in the test set

The selected checkpoint, e.g. `checkpoints/drugflow.ckpt`, must be specified in `configs/sampling/sample_and_maybe_evaluate.yml`.
To sample with your own trained model, simply provide a custom checkpoint path instead.

Furthermore, you need to update the `sample_outdir` parameter in the sampling config file or link the desired output location
```bash
SAMPLE_OUTDIR=...  # where samples will be saved
ln -s $SAMPLE_OUTDIR samples
```

For sampling, run
```bash
python src/sample_and_evaluate.py --config configs/sampling/sample_and_maybe_evaluate.yml
```
which supports parallelization across target pockets by specifying `--job_id` and `--n_jobs`.
To also evaluate the results, set `evaluate: True` in the sampling config file.

### Evaluating samples

We provide evaluators for metrics used in our paper. To evaluate samples, specify:

```bash
SAMPLES_DIR=...  # Location where the sampled dataset is stored
EVALUATED_DATA_ALL=...  # Temporary directory for evaluation output
EVALUATED_DATA=...  # Evaluation output
```

Run the evaluation:
```bash
python scripts/python/evaluate_baselines.py \
       --in_dir $SAMPLES_DIR \
       --out_dir $EVALUATED_DATA_ALL

python scripts/python/postprocess_metrics.py \
       --in_dir $EVALUATED_DATA_ALL \
       --out_dir $EVALUATED_DATA
```

Per-sample evaluation results will be stored in ```EVALUATED_DATA/metrics_detailed.csv``` and aggregated metrics will be stored in ```EVALUATED_DATA/metrics_aggregated.csv```.

## Samples

DrugFlow and baseline samples are available on [Zenodo](https://zenodo.org/records/14919171):
- [DrugFlow](https://zenodo.org/records/14919171/files/samples_drugflow.tar.gz?download=1)
- [TargetDiff](https://zenodo.org/records/14919171/files/samples_targetdiff.tar.gz?download=1)
- [DiffSBDD](https://zenodo.org/records/14919171/files/samples_diffsbdd.tar.gz?download=1)
- [Pocket2Mol](https://zenodo.org/records/14919171/files/samples_pocket2mol.tar.gz?download=1)

## Reference

```bibtex
@article{
  schneuing2025multidomain,
  title={Multi-domain distribution learning for de novo drug design},
  author={Schneuing, Arne and Igashov, Ilia and Dobbelstein, Adrian W and Castiglione, Thomas and Bronstein, Michael and Correia, Bruno},
  journal={arXiv preprint arXiv:2508.17815},
  year={2025}
}
```

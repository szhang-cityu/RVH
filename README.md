# RVH

This is the main implementation code of RVH/D-Researcher, which includes the diagnosis model training for generating diagnosis conclusion for comprehensive RVH/D source data.

## Environment

```bash
conda create -n rvh python=3.10
conda activate rvh
pip install numpy pandas torch scikit-learn matplotlib
pip install mamba-ssm
```

## Run training

```bash
bash run_train.sh
```

Then you can run the RL training on generating conclusion from comphrehensive ECG text record using the files in `train/train_LLM`:
```bash
cd train/train_LLM
bash run.sh
bash grpo.sh
bash report.sh
```

# Training & Monitoring

## Start Training

```bash
cd src
python train.py --config configs/report_3_2/td3_pink.yaml
#
```

## View TensorBoard (in seperate terminal)

```bash
cd src
tensorboard --logdir=logs/td3/checkpoint1_lunarlander_td3/tensorboard
```

Then open: http://localhost:6006

# Evaluation of trained agent

## Withour rendering

```bash
cr src
python evaluate.py --checkpoint logs/td3/checkpoint1_lunarlander_td3/checkpoints/td3_checkpoint_100000.pth --config configs/checkpoint1_lunarlander.yaml --episodes 20
```

```bash
cd src
python evaluate.py --checkpoint logs/td3/checkpoint1_lunarlander_td3/checkpoints/td3_checkpoint_100000.pth --config configs/checkpoint1_lunarlander.yaml --episodes 5 --render
```

```bash
python evaluate.py \
  --agent logs/td3/checkpoint3_hockey_strong_td3_pink/agent_final.pth \
          logs/td3/checkpoint3_hockey_strong_td3/agent_final.pth \
  --config configs/td3/checkpoint3_hockey_strong_pink.yaml \
           configs/td3/checkpoint3_hockey_strong.yaml \
  --name td3_pink td3_no_pink \
  --episodes 200 \
  --tag ablation_noise

```

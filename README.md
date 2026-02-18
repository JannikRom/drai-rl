# Training & Monitoring

## Start Training

```bash
cd src
python train.py --config configs/td3/checkpoint1_pendulum.yaml
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

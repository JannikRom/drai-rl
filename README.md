# Training & Monitoring

## Start Training

```bash
cd src
python train.py --config configs/checkpoint1_pendulum.yaml
#
```

## View TensorBoard (in seperate terminal)

```bash
cd src
tensorboard --logdir=logs/td3/checkpoint1_lunarlander_td3/tensorboard
```

Then open: http://localhost:6006

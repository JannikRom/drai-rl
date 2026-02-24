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

# Run Agent on the competition server

  1. Prerequisites
    Make sure your set_up_competition.sh contains your credentials:

    ```bash
    export COMPRL_SERVER_URL=<URL>
    export COMPRL_SERVER_PORT=<PORT>
    export COMPRL_ACCESS_TOKEN=<YOUR_TOKEN>
    ```

  2. Start a Remote Session
    Connect to the cluster and navigate to your project folder:

    ```bash
    ssh $remote_machine
    cd drai-rl/src
    ```

  3. Run Agent (without tmux)

    1. Load environment: set_up_competition.sh
    2. Start Agent:

      ```bash
        bash autorestart.sh --args --agent=drai_sac
      ```

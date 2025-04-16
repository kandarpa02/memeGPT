### Create a yaml config to train the model smoothly
example of how the config should look like:
    training:
        optimizer: 'AdamW'
        lr: 1e-5
        alpha: 0.99
        betas: [0.9, 0.999]
        weight_decay: 0.01
        momentum: 0.9
        epochs: 5
        load_weights: False
        weights_path: None
        precision: True
        num: 0
        ln: 0
        wte: 0
        wpe: 0




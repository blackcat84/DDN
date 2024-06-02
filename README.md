# DDN
Code of paper "A Doubly Decoupled Network for Edge Detection"

## Data 
BSDS500 following UAED  
NYUD follow PiDiNet  
BIPED following DexiNed  

## Running
```
python train_gslb.py --stepsize 3 --gpu 3 --output DDN-bsds --sampling 500 --dataset BSDS --encoder DDN-M36 --model model.sigma_logit_unet --batch_size 4 --note None --maxepoch 12 --kl_weight 0.01 -p 2000
```


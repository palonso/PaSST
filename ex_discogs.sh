CUDA_VISIBLE_DEVICES=0,1,2,3 \
    DDP=4 \
    python \
    ex_discogs.py with trainer.precision=16 models.net.arch=passt_s_swa_p16_128_ap476 -p -F output/discogs/ -c "PaSST base 4 GPU"

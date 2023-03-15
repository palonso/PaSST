CUDA_VISIBLE_DEVICES=0 python ex_mtt.py with \
    trainer.precision=16 \
    models.net.arch=passt_s_swa_p16_128_ap476_discogs \
    -p \
    -F output/mtt/ \
    -c "PaSST base 4 GPU"

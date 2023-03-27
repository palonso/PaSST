for n_block in 6
do
    CUDA_VISIBLE_DEVICES=0 python ex_discogs.py \
        extract_embeddings with \
        trainer.precision=16 \
        basedataset.predict_groundtruth="mtt/groundtruth-all.pk" \
        basedataset.base_dir="/home/palonso/data/magnatagatune-melspectrograms/" \
        norm_conf.norm_mean=1.5880631462493773 \
        norm_conf.norm_std=1.1815654825219488 \
        passt_discogs_30sec \
        inference.n_block=$n_block \
        passt_discogs30sec_inference
done

    # models.net.checkpoint="/home/palonso/reps/PaSST/output/discogs/230323-020758/discogs/d42e938b9d2d4d389c555c2f508fe277/checkpoints/epoch=129-step=541709.ckpt" \

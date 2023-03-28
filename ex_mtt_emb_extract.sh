for secs in 10 20 30
do

    CUDA_VISIBLE_DEVICES=0 python ex_discogs.py \
        extract_embeddings with \
        trainer.precision=16 \
        basedataset.predict_groundtruth="mtt/groundtruth-all.pk" \
        basedataset.base_dir="/home/palonso/data/magnatagatune-melspectrograms/" \
        norm_conf.norm_mean=1.5880631462493773 \
        norm_conf.norm_std=1.1815654825219488 \
        inference.n_block=6 \
        passt_discogs_"$secs"sec \
        passt_discogs"$secs"sec_inference

done

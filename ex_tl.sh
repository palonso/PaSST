for weight_decay in 1e-3 1e-4 1e-5
do
    for batch_size in 64 128 256
    do
        for max_lr in 1e-4 1e-3 1e-2
        do
            CUDA_AVAILABLE_DEVICES=0 python3 ex_tl.py with \
                data.base_dir=embeddings/mtt/30sec/swa/6/ \
                data.types=cdt \
                model.scheduler=cyclic \
                model.base_lr=1e-5 \
                model.max_lr=$max_lr \
                model.weight_decay=$weight_decay \
                data.batch_size=$batch_size
        done
    done
done

    # models.net.checkpoint="/home/palonso/reps/PaSST/output/discogs/230323-020758/discogs/d42e938b9d2d4d389c555c2f508fe277/checkpoints/epoch=129-step=541709.ckpt" \

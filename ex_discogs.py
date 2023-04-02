import comet_ml

import matplotlib as mpl
mpl.use('Agg')

import os
from collections import defaultdict
from itertools import chain
from pathlib import Path
from datetime import datetime
import sys

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger, CSVLogger
from sacred.config_helpers import DynamicIngredient, CMD
from torch.nn import functional as F
import numpy as np

from ba3l.experiment import Experiment
from ba3l.module import Ba3lModule

from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from config_updates import add_configs
from helpers.mixup import my_mixup
from helpers.models_size import count_non_zero_params
from helpers.ramp import exp_warmup_linear_down, cosine_cycle
from helpers.workersinit import worker_init_fn
from helpers.spec_masking import SpecMasking
from sklearn import metrics

ex = Experiment("discogs")

# Example call with all the default config:
# python ex_discogs.py with trainer.precision=16 models.net.arch=passt_deit_bd_p16_384 -p -m mongodb_server:27000:discogs -c "PaSST base"
# with 2 gpus:
# DDP=2 python ex_discogs.py with trainer.precision=16  models.net.arch=passt_deit_bd_p16_384 -p -m mongodb_server:27000:discogs -c "PaSST base 2 GPU"

# define datasets and loaders
ex.datasets.training.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn), train=True, batch_size=12,
                          num_workers=16, shuffle=None, dataset=CMD("/basedataset.get_train_set"),
                          sampler=CMD("/basedataset.get_ft_weighted_sampler"))

get_validate_loader = ex.datasets.val.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn),
                                            validate=True, batch_size=20, num_workers=16,
                                            dataset=CMD("/basedataset.get_val_set"))


@ex.config
def default_conf():
    cmd = " ".join(sys.argv) # command line arguments
    saque_cmd = os.environ.get("SAQUE_CMD", "").strip()
    saque_id = os.environ.get("SAQUE_ID", "").strip()
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    if os.environ.get("SLURM_ARRAY_JOB_ID", False):
        slurm_job_id = os.environ.get("SLURM_ARRAY_JOB_ID", "").strip() + "_" + os.environ.get("SLURM_ARRAY_TASK_ID",
                                                                                               "").strip()
    process_id = os.getpid()
    models = {
        "net": DynamicIngredient(
            "models.passt.model_ing",
            arch="passt_deit_bd_p16_384",
            n_classes=400,
            s_patchout_t=30,
            s_patchout_f=3,
            input_fdim=96,
            input_tdim=625,
            use_swa=True,
        ),  # network config
        "mel": DynamicIngredient(
            "models.preprocess.model_ing",
            instance_cmd="AugmentMelSTFT",
            n_mels=96,
            sr=16000,
            win_length=512,
            hopsize=256,
            n_fft=512,
            # freqm=48,
            # timem=192,
            # htk=False,
            fmin=0.0,
            fmax=None,
            norm=1,
            fmin_aug_range=10,
            fmax_aug_range=2000,
        )
    }
    basedataset = DynamicIngredient(
        "discogs.dataset.dataset",
        wavmix=1,
        clip_length=10,
    )
    trainer = dict(
        max_epochs=130,
        gpus=1,
        weights_summary='full',
        benchmark=True,
        num_sanity_val_steps=0,
        reload_dataloaders_every_epoch=True,
        sync_batchnorm=True,
        )
    lr = 0.00002 # learning rate
    use_mixup = True
    use_masking = True
    mixup_alpha = 0.3
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")


# register extra possible configs
add_configs(ex)


@ex.command
def get_scheduler_lambda(warm_up_len=5, ramp_down_start=50, ramp_down_len=50, last_lr_value=0.01,
                         schedule_mode="exp_lin"):
    if schedule_mode == "exp_lin":
        return exp_warmup_linear_down(warm_up_len, ramp_down_len, ramp_down_start, last_lr_value)
    if schedule_mode == "cos_cyc":
        return cosine_cycle(warm_up_len, ramp_down_start, last_lr_value)
    raise RuntimeError(f"schedule_mode={schedule_mode} Unknown for a lambda funtion.")


@ex.command
def get_lr_scheduler(optimizer, schedule_mode):
    if schedule_mode in {"exp_lin", "cos_cyc"}:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, get_scheduler_lambda())
    raise RuntimeError(f"schedule_mode={schedule_mode} Unknown.")


@ex.command
def get_optimizer(params, lr, adamw=True, weight_decay=0.0001):
    if adamw:
        print(f"\nUsing adamw weight_decay={weight_decay}!\n")
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.Adam(params, lr=lr)


class M(Ba3lModule):
    def __init__(self, experiment):
        self.mel = None
        self.da_net = None
        super(M, self).__init__(experiment)

        self.use_mixup = self.config.use_mixup or False
        self.use_masking = self.config.use_masking or False
        self.mixup_alpha = self.config.mixup_alpha

        desc, sum_params, sum_non_zero = count_non_zero_params(self.net)
        self.experiment.info["start_sum_params"] = sum_params
        self.experiment.info["start_sum_params_non_zero"] = sum_non_zero

        # in case we need embedings for the DA
        self.net.return_embed = True
        self.dyn_norm = self.config.dyn_norm
        self.do_swa = True

        self.distributed_mode = self.config.trainer.num_nodes > 1

        if self.use_masking:
            self.masking = SpecMasking()

    def forward(self, x):
        return self.net(x)

    def mel_forward(self, x):
        # input is already mel_spec
        # old_shape = x.size()
        # x = x.reshape(-1, old_shape[2])
        # x = self.mel(x)
        # x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
        if self.dyn_norm:
            if not hasattr(self, "tr_m") or not hasattr(self, "tr_std"):
                tr_m, tr_std = get_dynamic_norm(self)
                self.register_buffer('tr_m', tr_m)
                self.register_buffer('tr_std', tr_std)
            x = (x - self.tr_m) / self.tr_std
        return x

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, f, y = batch

        if self.mel:
            x = self.mel_forward(x)

        batch_size = len(y)

        rn_indices, lam = None, None
        if self.use_mixup:
            rn_indices, lam = my_mixup(batch_size, self.mixup_alpha)
            lam = lam.to(x.device)
            x = x * lam.reshape(batch_size, 1, 1, 1) + x[rn_indices] * (1. - lam.reshape(batch_size, 1, 1, 1))

        if self.use_masking:
            x = self.masking.compute(x)

        for i in range(len(x)):
            if not Path(f"example{i}.png").exists():
                patch = x[i].detach().cpu().numpy().squeeze()
                plt.matshow(patch, aspect="auto")
                plt.colorbar()
                plt.savefig(f"example{i}.png")

        y_hat, embed = self.forward(x)

        if self.use_mixup:
            y_mix = y * lam.reshape(batch_size, 1) + y[rn_indices] * (1. - lam.reshape(batch_size, 1))
            samples_loss = F.binary_cross_entropy_with_logits(
                y_hat, y_mix, reduction="none")
            loss = samples_loss.mean()
        else:
            samples_loss = F.binary_cross_entropy_with_logits(y_hat, y, reduction="none")
            loss = samples_loss.mean()

        results = {"loss": loss, }

        return results

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        logs = {'train.loss': avg_loss, 'step': self.current_epoch}

        self.log_dict(logs, sync_dist=True)

    def predict(self, batch, batch_idx: int, dataloader_idx: int = None):
        x, f, y = batch
        if self.mel:
            x = self.mel_forward(x)

        embeddings = self.net.forward_until_block(
            x,
            n_block=self.config.inference.n_block,
            return_self_attention=False,
            compact_features=True,
        )

        results = {"embeddings": embeddings.detach()}
        results = {k: v.cpu() for k, v in results.items()}
        results["filename"] = f
        return results

    def validation_step(self, batch, batch_idx, stage="val"):
        x, f, y = batch
        if self.mel:
            x = self.mel_forward(x)

        results = {}
        model_name = [("", self.net)]
        if self.do_swa:
            model_name = model_name + [("swa_", self.net_swa)]
        for net_name, net in model_name:
            y_hat, _ = net(x)
            samples_loss = F.binary_cross_entropy_with_logits(y_hat, y)
            loss = samples_loss.mean()
            out = torch.sigmoid(y_hat.detach())
            # self.log("validation.loss", loss, prog_bar=True, on_epoch=True, on_step=False)
            results = {**results, net_name + f"{stage}_loss": loss, net_name + "out": out, net_name + "target": y.detach()}
        results = {k: v.cpu() for k, v in results.items()}
        return results

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, stage="test")

    def validation_epoch_end(self, outputs, stage="val"):
        model_name = [("", self.net)]
        if self.do_swa:
            model_name = model_name + [("swa_", self.net_swa)]
        for net_name, net in model_name:
            avg_loss = torch.stack([x[net_name + f'{stage}_loss'] for x in outputs]).mean()
            out = torch.cat([x[net_name + 'out'] for x in outputs], dim=0)
            target = torch.cat([x[net_name + 'target'] for x in outputs], dim=0)

            if not self.distributed_mode:
                try:
                    average_precision = metrics.average_precision_score(
                        target.float().numpy(), out.float().numpy(), average=None)
                except ValueError:
                    average_precision = np.array([np.nan] * self.net.n_classes)
                try:
                    roc = metrics.roc_auc_score(target.numpy(), out.numpy(), average=None)
                except ValueError:
                    roc = np.array([np.nan] * self.net.n_classes)
                logs = {net_name + f'{stage}_loss': torch.as_tensor(avg_loss).cuda(),
                        net_name + f'{stage}_ap': torch.as_tensor(average_precision.mean()).cuda(),
                        net_name + f'{stage}_roc': torch.as_tensor(roc.mean()).cuda(),
                        'step': torch.as_tensor(self.current_epoch).cuda()}
                self.log_dict(logs)

            if self.distributed_mode:
                allout = self.all_gather(out)
                alltarget = self.all_gather(target)
                alltarget = alltarget.reshape(-1, alltarget.shape[-1]).cpu().numpy()
                allout = allout.reshape(-1, allout.shape[-1]).cpu().numpy()

                average_precision = metrics.average_precision_score(alltarget, allout, average=None)
                roc = metrics.roc_auc_score(alltarget, allout, average=None)
                if self.trainer.is_global_zero:
                    logs = {
                        net_name + f"{stage}_ap": torch.as_tensor(average_precision.mean()).cuda(),
                        net_name + f"{stage}_roc": torch.as_tensor(roc.mean()).cuda(),
                        'step': torch.as_tensor(self.current_epoch).cuda()
                    }
                    self.log_dict(logs, sync_dist=False)

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs, stage="test")

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        optimizer = get_optimizer(self.parameters())
        # torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': get_lr_scheduler(optimizer)
        }

    def configure_callbacks(self):
        return get_extra_checkpoint_callback() + get_extra_swa_callback()

    def predict_dataloader(self):
        from discogs.dataset import get_predict_set
        return DataLoader(
            get_predict_set(),
            batch_size=24,
            num_workers=16,
            )

    def test_dataloader(self):
        from discogs.dataset import get_test_set
        return DataLoader(
            get_test_set(),
            batch_size=24,
            num_workers=16,
            )

@ex.command
def get_dynamic_norm(model, dyn_norm=False):
    if not dyn_norm:
        return None, None
    raise RuntimeError('no dynamic norm supported yet.')


@ex.command
def get_extra_checkpoint_callback(save_last_n=None):
    if save_last_n is None:
        return []
    return [ModelCheckpoint(monitor="step", verbose=True, save_top_k=save_last_n, mode='max')]


@ex.command
def get_extra_swa_callback(swa=True, swa_epoch_start=50,
                           swa_freq=5):
    if not swa:
        return []
    print("\n Using swa!\n")
    from helpers.swa_callback import StochasticWeightAveraging
    return [StochasticWeightAveraging(swa_epoch_start=swa_epoch_start, swa_freq=swa_freq)]


@ex.command
def main(_run, _config, _log, _rnd, _seed):
    trainer = ex.get_trainer()
    train_loader = ex.get_train_dataloaders()
    val_loader = ex.get_val_dataloaders()

    modul = M(ex)

    if os.environ["NODE_RANK"] == "0":
        project_name = _config["basedataset"]["name"]
        save_dir = Path("output", project_name, _config["timestamp"])
        comet_logger = CometLogger(
            project_name=project_name,
            api_key=os.environ["COMET_API_KEY"],
            save_dir=save_dir,
            experiment_name=_config["timestamp"],
            )
        csv_logger = CSVLogger(save_dir=save_dir)
        trainer.logger_connector.on_trainer_init([comet_logger, csv_logger], 200, 200, False)
        trainer.logger.log_hyperparams(_config)

    trainer.fit(
        modul,
        train_dataloader=train_loader,
        val_dataloaders=val_loader,
    )

    return {"done": True}


@ex.command
def model_speed_test(_run, _config, _log, _rnd, _seed, speed_test_batch_size=100):
    '''
    Test training speed of a model
    @param _run:
    @param _config:
    @param _log:
    @param _rnd:
    @param _seed:
    @param speed_test_batch_size: the batch size during the test
    @return:
    '''

    modul = M(ex)
    modul = modul.cuda()
    batch_size = speed_test_batch_size
    print(f"\nBATCH SIZE : {batch_size}\n")
    test_length = 100
    print(f"\ntest_length : {test_length}\n")

    x = torch.ones([batch_size, 1, 128, 998]).cuda()
    target = torch.ones([batch_size, 400]).cuda()
    # one passe
    net = modul.net
    # net(x)
    scaler = torch.cuda.amp.GradScaler()
    torch.backends.cudnn.benchmark = True
    # net = torch.jit.trace(net,(x,))
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

    print("warmup")
    import time
    torch.cuda.synchronize()
    t1 = time.time()
    for i in range(10):
        with  torch.cuda.amp.autocast():
            y_hat, embed = net(x)
            loss = F.binary_cross_entropy_with_logits(y_hat, target, reduction="none").mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    t2 = time.time()
    print('warmup done:', (t2 - t1))
    torch.cuda.synchronize()
    t1 = time.time()
    print("testing speed")

    for i in range(test_length):
        with  torch.cuda.amp.autocast():
            y_hat, embed = net(x)
            loss = F.binary_cross_entropy_with_logits(y_hat, target, reduction="none").mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    t2 = time.time()
    print('test done:', (t2 - t1))
    print("average speed: ", (test_length * batch_size) / (t2 - t1), " specs/second")


@ex.command
def evaluate_only(_run, _config, _log, _rnd, _seed):
    # force overriding the config, not logged = not recommended
    trainer = ex.get_trainer()
    train_loader = ex.get_train_dataloaders()
    val_loader = ex.get_val_dataloaders()
    modul = M(ex)
    modul.val_dataloader = None
    trainer.val_dataloaders = None
    print(f"\n\nValidation len={len(val_loader)}\n")
    res = trainer.validate(modul, val_dataloaders=val_loader)
    print("\n\n Validtaion:")
    print(res)

@ex.command
def extract_embeddings(_run, _config, _log, _rnd, _seed):

    trainer = ex.get_trainer()
    modul = M(ex)
    modul.eval()

    outputs = trainer.predict(modul)

    filenames = list(chain.from_iterable([x['filename'] for x in outputs]))
    print("n filenames:", len(filenames))
    for output in ["embeddings"]:
        print("processing output", output)
        out = np.vstack([x[output] for x in outputs])

        print(f"n {output}:", len(out))

        agg_out = defaultdict(list)
        for o, f in zip(out, filenames):
            agg_out[f].append(o)

        agg_out = {k: np.array(o) for k, o in agg_out.items()}
        subdir1 = str(_config["basedataset"]["clip_length"]) + "sec"
        subdir2 = "swa" if _config["models"]["net"]["use_swa"] else "no_swa"
        if _config["models"]["net"]["s_patchout_f_indices"]:
            removed_bands = "_".join(np.array(_config["models"]["net"]["s_patchout_f_indices"]).astype("str"))
            subdir2 += f"_patchout_f_indices" + removed_bands
        if _config["models"]["net"]["s_patchout_t_indices"]:
            removed_bands = "_".join(np.array(_config["models"]["net"]["s_patchout_t_indices"]).astype("str"))
            subdir2 += f"_patchout_f_forced_" + removed_bands
        if _config["models"]["net"]["s_patchout_f_interleaved"]:
            subdir2 += f"_patchout_f_interleaved" + str(_config["models"]["net"]["s_patchout_f_interleaved"])
        if _config["models"]["net"]["s_patchout_t_interleaved"]:
            subdir2 += f"_patchout_t_interleaved" + str(_config["models"]["net"]["s_patchout_t_interleaved"])
        subdir3 = str(_config["inference"]["n_block"])
        out_dir = Path(_config["inference"]["out_dir"]) / subdir1 / subdir2 / subdir3

        for k, v in agg_out.items():
            file_path = out_dir / (k + f".{output}")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(file_path, v)

@ex.command
def test(_config):
    trainer = ex.get_trainer()
    modul = M(ex)
    modul.eval()

    if os.environ["NODE_RANK"] == "0":
        project_name = "discogs_test"
        trainer.logger = CometLogger(
            project_name=project_name,
            api_key=os.environ["COMET_API_KEY"],
        )
        trainer.logger.log_hyperparams(_config)

    trainer.test(modul)

@ex.command
def compute_norm_stats(_run, _config, _log, _rnd, _seed):
    # force overriding the config, not logged = not recommended
    loader = ex.get_train_dataloaders()
    mean = []
    std = []

    for i, (audio_input, _, _) in tqdm(enumerate(loader), total=len(loader)):
        audio_input = audio_input.type(torch.DoubleTensor)
        cur_mean = torch.mean(audio_input)
        cur_std = torch.std(audio_input)
        mean.append(cur_mean)
        std.append(cur_std)
        # print(cur_mean, cur_std, np.max(audio_input), np.min(audio_input))
    print(np.mean(mean), np.mean(std))

@ex.command
def test_loaders():
    '''
    get one sample from each loader for debbuging
    @return:
    '''
    for i, b in enumerate(ex.datasets.training.get_iter()):
        print(b)
        break

    for i, b in enumerate(ex.datasets.test.get_iter()):
        print(b)
        break


def set_default_json_pickle(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


def multiprocessing_run(rank, word_size):
    print("rank ", rank, os.getpid())
    print("word_size ", word_size)
    os.environ['NODE_RANK'] = str(rank)
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES'].split(",")[rank]
    argv = sys.argv
    if rank != 0:
        print(f"Unobserved {os.getpid()} with rank {rank}")
        argv = argv + ["-u"]  # only rank 0 is observed
    if "with" not in argv:
        argv = argv + ["with"]

    argv = argv + [f"trainer.num_nodes={word_size}", f"trainer.accelerator=ddp"]
    print(argv)

    @ex.main
    def default_command():
        return main()

    ex.run_commandline(argv)


if __name__ == '__main__':
    # set DDP=2 forks two processes to run on two GPUs
    # the environment variable "DDP" define the number of processes to fork
    # With two 2x 2080ti you can train the full model to .47 in around 24 hours
    # you may need to set NCCL_P2P_DISABLE=1
    word_size = os.environ.get("DDP", None)
    if word_size:
        import random

        word_size = int(word_size)
        print(f"\n\nDDP TRAINING WITH WORD_SIZE={word_size}\n\n")
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = f"{9999 + random.randint(0, 9999)}"  # plz no collisions
        os.environ['PL_IN_DDP_SUBPROCESS'] = '1'

        for rank in range(word_size):
            pid = os.fork()
            if pid == 0:
                print("Child Forked ")
                multiprocessing_run(rank, word_size)
                exit(0)

        pid, exit_code = os.wait()
        print(pid, exit_code)
        exit(0)

print("__main__ is running pid", os.getpid(), "in module main: ", __name__)


@ex.automain
def default_command():
    return main()

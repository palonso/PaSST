import comet_ml

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import logging
import os
import sys
from collections import defaultdict
from itertools import chain
from datetime import datetime
from pathlib import Path

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from sacred.config_helpers import DynamicIngredient, CMD
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np

from ba3l.experiment import Experiment
from ba3l.module import Ba3lModule

from torch.utils.data import DataLoader

from config_updates import add_configs
from helpers.mixup import my_mixup
from helpers.models_size import count_non_zero_params
from helpers.ramp import exp_warmup_linear_down, cosine_cycle
from helpers.workersinit import worker_init_fn
from helpers.spec_masking import SpecMasking
from sklearn import metrics

logger = logging.getLogger()
logger.setLevel(logging.WARNING)


ex = Experiment("mtt")

# Example call with all the default config:
# python ex_mtt.py with trainer.precision=16 models.net.arch=passt_deit_bd_p16_384 -p -m mongodb_server:27000:mtt -c "PaSST base"
# with 2 gpus:
# DDP=2 python ex_mtt.py with trainer.precision=16  models.net.arch=passt_deit_bd_p16_384 -p -m mongodb_server:27000:mtt -c "PaSST base 2 GPU"

# define datasets and loaders
ex.datasets.training.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn), train=True, batch_size=28,
                          num_workers=16, shuffle=None, dataset=CMD("/basedataset.get_train_set"))

get_validate_loader = ex.datasets.val.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn),
                                            validate=True, batch_size=32, num_workers=16,
                                            dataset=CMD("/basedataset.get_val_set"))

get_test_loader = ex.datasets.test.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn),
                                        validate=True, batch_size=32, num_workers=16,
                                        dataset=CMD("/basedataset.get_test_set"))

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
            n_classes=50,
            s_patchout_t=0,
            s_patchout_f=0,
            input_fdim=96,
            input_tdim=625,
            checkpoint="",
            remove_n_blocks=0,
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
            fmax_aug_range=2000
        )
    }
    basedataset = DynamicIngredient("mtt.dataset.dataset", wavmix=1)
    trainer = dict(
        max_epochs=30,
        gpus=1,
        weights_summary='full',
        benchmark=True,
        num_sanity_val_steps=0,
        reload_dataloaders_every_epoch=True
    )
    lr = 1e-4 # learning rate
    lr_mult = 0.3
    use_mixup = False
    use_masking = True
    freq_masks = 10
    time_masks = 20
    mixup_alpha = 0.3

    schedule_mode = "exp_lin"
    warm_up_len = 5
    cycle_len = 5
    ramp_down_start = 10
    ramp_down_len = 30
    last_lr_value = 1e-6
    weight_decay=  0.0001
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")

# register extra possible configs
add_configs(ex)


@ex.command
def get_scheduler_lambda(warm_up_len=5, cycle_len=5, ramp_down_start=50, ramp_down_len=50, last_lr_value=0.01,
                         schedule_mode="exp_lin"):
    if schedule_mode == "exp_lin":
        return exp_warmup_linear_down(warm_up_len, ramp_down_len, ramp_down_start, last_lr_value)
    if schedule_mode == "cos_cyc":
        return cosine_cycle(cycle_len, ramp_down_start, last_lr_value)
    raise RuntimeError(f"schedule_mode={schedule_mode} Unknown for a lambda function.")


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

        # in case we need embeddings for the DA
        self.net.return_embed = True
        self.dyn_norm = self.config.dyn_norm
        self.do_swa = False

        self.distributed_mode = self.config.trainer.num_nodes > 1

        if self.use_masking:
            self.masking = SpecMasking(
                freq_masks=self.config.freq_masks,
                time_masks=self.config.time_masks,
            )

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


        y_hat, embed = self.forward(x)

        if self.use_mixup:
            y_mix = y * lam.reshape(batch_size, 1) + y[rn_indices] * (1. - lam.reshape(batch_size, 1))
            samples_loss = F.binary_cross_entropy_with_logits(
                y_hat, y_mix, reduction="none")
            loss = samples_loss.mean()
            samples_loss = samples_loss.detach()
        else:
            samples_loss = F.binary_cross_entropy_with_logits(y_hat, y, reduction="none")
            loss = samples_loss.mean()
            samples_loss = samples_loss.detach()

        self.log("train_loss", loss, on_epoch=True)

        return loss

    # def training_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

    #     logs = {'train_loss': avg_loss,
    #             # 'step': self.current_epoch
    #         }

        # self.log_dict(logs, sync_dist=True)

    def predict(self, batch, batch_idx: int, dataloader_idx: int = None):
        x, f, y = batch
        if self.mel:
            x = self.mel_forward(x)

        y_hat, _ = self.forward(x)
        return f, y_hat

    def validation_step(self, batch, batch_idx):
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
            results = {**results, net_name + "val_loss": loss, net_name + "out": out, net_name + "target": y.detach()}
        results = {k: v.cpu() for k, v in results.items()}
        self.log("val_loss", results["val_loss"], prog_bar=True)

        return results

    def validation_epoch_end(self, outputs):
        model_name = [("", self.net)]
        if self.do_swa:
            model_name = model_name + [("swa_", self.net_swa)]
        for net_name, net in model_name:
            avg_loss = torch.stack([x[net_name + 'val_loss'] for x in outputs]).mean()
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
                logs = {net_name + 'val_loss': torch.as_tensor(avg_loss).cuda(),
                        net_name + 'val_ap': torch.as_tensor(average_precision.mean()).cuda(),
                        net_name + 'val_roc': torch.as_tensor(roc.mean()).cuda(),
                        # 'step': torch.as_tensor(self.current_epoch).cuda()\
                    }
                # torch.save(average_precision, f"ap_perclass_{average_precision.mean()}.pt")
                # print(average_precision)
                self.log_dict(logs, sync_dist=True, on_epoch=True)

            if self.distributed_mode:
                allout = self.all_gather(out)
                alltarget = self.all_gather(target)
                alltarget = alltarget.reshape(-1, alltarget.shape[-1]).cpu().numpy()
                allout = allout.reshape(-1, allout.shape[-1]).cpu().numpy()

                average_precision = metrics.average_precision_score(alltarget, allout, average=None)
                roc = metrics.roc_auc_score(alltarget, allout, average=None)
                if self.trainer.is_global_zero:
                    logs = {
                        net_name + "val_ap": torch.as_tensor(average_precision.mean()).cuda(),
                        net_name + "val_roc": torch.as_tensor(roc.mean()).cuda(),
                        # 'step': torch.as_tensor(self.current_epoch).cuda()
                    }
                    self.log_dict(logs, sync_dist=True, on_epoch=True)
            # else:
            #     self.log_dict({net_name + "ap": logs[net_name + 'ap'], 'step': logs['step']}, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, f, y = batch
        if self.mel:
            x = self.mel_forward(x)

        results = {}
        model_name = [("", self.net)]
        for net_name, net in model_name:
            y_hat, _ = net(x)
            samples_loss = F.binary_cross_entropy_with_logits(y_hat, y)
            loss = samples_loss.mean()
            out = torch.sigmoid(y_hat.detach())
            results = {
                **results,
                net_name + "test_loss": loss,
                net_name + "out": out,
                net_name + "target": y.detach(),
                }
        results = {k: v.cpu() for k, v in results.items()}
        results[net_name + "filename"] = f

        return results

    def test_epoch_end(self, outputs):
        model_name = [("", self.net)]
        if self.do_swa:
            model_name = model_name + [("swa_", self.net_swa)]
        for net_name, net in model_name:
            avg_loss = torch.stack([x[net_name + 'test_loss'] for x in outputs]).mean()
            out = torch.cat([x[net_name + 'out'] for x in outputs], dim=0)
            target = torch.cat([x[net_name + 'target'] for x in outputs], dim=0)
            filenames = list(chain.from_iterable([x[net_name + 'filename'] for x in outputs]))

            agg_out = defaultdict(list)
            agg_target = defaultdict(list)
            for o, t, f in zip(out.float().numpy(), target.float().numpy(), filenames):
                agg_out[f].append(o)
                agg_target[f].append(t)
            
            agg_out = np.array([np.mean(o, axis=0) for o in agg_out.values()])
            agg_target = np.array([np.mean(o, axis=0) for o in agg_target.values()])

            try:
                average_precision = metrics.average_precision_score(
                    agg_target, agg_out, average=None)
            except ValueError:
                average_precision = np.array([np.nan] * self.net.n_classes)
            try:
                roc = metrics.roc_auc_score(agg_target, agg_out, average=None)
            except ValueError:
                roc = np.array([np.nan] * self.net.n_classes)
            logs = {net_name + 'test_loss': torch.as_tensor(avg_loss).cuda(),
                    net_name + 'test_ap': torch.as_tensor(average_precision.mean()).cuda(),
                    net_name + 'test_roc': torch.as_tensor(roc.mean()).cuda(),
                    # 'step': torch.as_tensor(self.current_epoch).cuda()
            }
            self.log_dict(logs, sync_dist=True)

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        # torch.optim.Adam(self.parameters(), lr=self.config.lr)

        lr = self.config.lr
        lr_orig = lr
        lr_mult = self.config.lr_mult
        layer_names = []
        for idx, (name, param) in enumerate(self.net.named_parameters()):
            layer_names.append(name)
 
        layer_names.reverse()

        parameters = []
        # store params & learning rates
        nb = 11 - self.config.models.net.remove_n_blocks
        next_block = f"blocks.{nb}"
        for idx, name in enumerate(layer_names):
            # update learning rate
            if name.startswith(next_block):
                bn = int(name.split(".")[1])
                next_block = f"blocks.{bn - 1}"

                lr *= lr_mult

            # display info
            print(f'{idx}: lr = {lr_orig if name in ("dist_token", "cls_token") else lr:.9f}, {name}')
            # append layer parameters
            parameters += [{'params': [p for n, p in self.net.named_parameters() if n == name and p.requires_grad],
    'lr': lr_orig if name in ("dist_token", "cls_token") else lr}]

        optimizer = get_optimizer(parameters)

        return {
            'optimizer': optimizer,
            'lr_scheduler': get_lr_scheduler(optimizer)
        }

    def configure_callbacks(self):
        # return get_extra_checkpoint_callback() + get_extra_swa_callback()
        return get_extra_checkpoint_callback()


@ex.command
def get_dynamic_norm(model, dyn_norm=False):
    if not dyn_norm:
        return None, None
    raise RuntimeError('no dynamic norm supported yet.')


@ex.command
def get_extra_checkpoint_callback(save_last_n=None):
    # if save_last_n is None:
    #     return []
    return [ModelCheckpoint(
        filename="max_roc",
        save_top_k=1,
        monitor="val_roc",
        mode='max',
        save_weights_only=True,
        verbose=True,
        every_n_train_steps=0,
    )]


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
    val_loaders = ex.get_val_dataloaders()

    modul = M(ex)

    project_name = _config["basedataset"]["name"]
    comet_logger = CometLogger(
        project_name=project_name,
        api_key=os.environ["COMET_API_KEY"],
        save_dir = Path("output", project_name, _config["timestamp"])
        )
    trainer.logger = comet_logger
    comet_logger.log_hyperparams(_config)

    trainer.fit(
        modul,
        train_dataloader=train_loader,
        val_dataloaders=val_loaders["val"],
    )

    trainer.test(
        model=modul,
        test_dataloaders=val_loaders["test"],
        ckpt_path="max_roc",
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
    target = torch.ones([batch_size, 50]).cuda()
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
    print("mean:", np.mean(mean))
    print("std:", np.mean(std))


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

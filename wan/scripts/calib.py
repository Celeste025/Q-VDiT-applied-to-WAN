"""Wan2.1 Q-VDiT calibration (PTQ + block reconstruction)."""

import argparse
import gc
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from diffusers import WanTransformer3DModel
from omegaconf import OmegaConf
from tqdm import trange

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdiff.models.quant_model import QuantModel
from qdiff.optimization.model_recon import our_model_reconstruction
from qdiff.utils import get_quant_calib_data
from wan.models.wan_forward_adapter import WanForwardAdapter
from wan.utils.config import parse_dtype

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--calib_config", required=True)
    p.add_argument("--calib_data", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--part_fp", action="store_true")
    p.add_argument("--time_mp_config_weight", default=None)
    p.add_argument("--time_mp_config_act", default=None)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    opt = parse_args()
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    if os.path.exists(os.path.join(outpath, "config.yaml")):
        os.remove(os.path.join(outpath, "config.yaml"))
    shutil.copy(opt.calib_config, os.path.join(outpath, "config.yaml"))
    if os.path.exists(os.path.join(outpath, "qdiff")):
        shutil.rmtree(os.path.join(outpath, "qdiff"))
    shutil.copytree(ROOT / "qdiff", os.path.join(outpath, "qdiff"))

    log_path = os.path.join(outpath, "run.log")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
    )
    config = OmegaConf.load(opt.calib_config)
    logger.info("Conducting Command: %s", " ".join(sys.argv))

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpus = [int(x) for x in opt.gpu.split(",")]
    torch.cuda.set_device(gpus[0])
    dtype = parse_dtype(opt.dtype)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)

    transformer = WanTransformer3DModel.from_pretrained(
        os.path.join(opt.model_path, "transformer"),
        torch_dtype=dtype,
    )
    model = WanForwardAdapter(transformer, cfg_split=config.get("cfg_split", True))
    model = model.to(device).eval()

    # ======================================================
    # 4. build quantized model
    # ======================================================
    # TODO: only feed part of the wq_params, since it is directly used for quantizer init
    wq_params = config.quant.weight.quantizer
    aq_params = config.quant.activation.quantizer
    # use_weight_quant = False if wq_params is None else True
    # use_act_quant = False if aq_params is None else True

    if config.get('mixed_precision', False):
        wq_params['mixed_precision'] = config.mixed_precision
        # aq_params['mixed_precision'] = config.mixed_precision
    if config.get('timestep_wise', False):
        aq_params['timestep_wise'] = config.timestep_wise
    if getattr(opt, 'smooth_quant_alpha', None):
        assert aq_params.smooth_quant.enable
        aq_params.smooth_quant.alpha = opt.smooth_quant_alpha
        with open(os.path.join(outpath, 'config.yaml'), 'r') as file:
            quant_config_content = yaml.safe_load(file)
        old_smooth_quant_alpha = quant_config_content['quant']['activation']['quantizer']['smooth_quant']['alpha']
        if isinstance(old_smooth_quant_alpha, list):
            new_smooth_quant_alpha = opt.smooth_quant_alpha
        else:
            new_smooth_quant_alpha = opt.smooth_quant_alpha[0]  # single value
        quant_config_content['quant']['activation']['quantizer']['smooth_quant']['alpha'] = new_smooth_quant_alpha
        with open(os.path.join(outpath, 'config.yaml'), 'w') as file:
            yaml.dump(quant_config_content, file, default_flow_style=False)
        logger.info(f"replacing the original alpha {old_smooth_quant_alpha} with the alpha {new_smooth_quant_alpha} given in --arg")

    qnn = QuantModel(
        model=model,
        weight_quant_params=wq_params,
        act_quant_params=aq_params,
        model_type=config.model.model_type,
    )
    qnn.cuda()
    qnn.eval()
    logger.info(qnn)

    # DIRTY: set the cfg_split as the attribute of the model
    # the cfg_split is configured in `opensora/schedulers/ippdm/__init__.py`
    cfg_split = config.get('cfg_split', False)
    qnn.cfg_split = cfg_split

    if not config.quant.grad_checkpoint:
        logger.info('Not use gradient checkpointing for transformer blocks')
        qnn.set_grad_ckpt(False)
    elif config.model.model_type == "wan":
        transformer = getattr(getattr(qnn.model, "model", None), "transformer", None)
        if transformer is not None and hasattr(transformer, "enable_gradient_checkpointing"):
            transformer.enable_gradient_checkpointing()
            logger.info("Enabled gradient checkpointing on Wan transformer (config.quant.grad_checkpoint=True)")

    logger.info(f"Sampling data from {config.calib_data.n_steps} timesteps for calibration")
    # INFO: feed the calib_data path through argparse also, overwrite quant config
    if hasattr(opt, "calib_data") and opt.calib_data is not None:
        config.calib_data.path = opt.calib_data
    calib_data_ckpt = torch.load(config.calib_data.path, map_location="cpu")
    calib_data = get_quant_calib_data(
        config,
        calib_data_ckpt,
        config.calib_data.n_steps,
        model_type=config.model.model_type,
        repeat_interleave=False,
    )
    del(calib_data_ckpt)
    gc.collect()

    # ======================================================
    # 5. prepare data for init the model
    # ======================================================
    calib_added_kwargs = {}
    if config.model.model_type == "wan":
        calib_batch_size = config.calib_data.batch_size*2  # INFO: used to support the CFG
        logger.info(f"Calibration data shape: {calib_data[0].shape} {calib_data[1].shape} {calib_data[2].shape}")
        calib_xs, calib_ts, calib_cs, calib_masks = calib_data
        calib_added_kwargs["mask"] = calib_masks
    else:
        raise NotImplementedError

    # ======================================================
    # 6. get the quant params (training-free), using the calibration data
    # ======================================================

    # for part quantization
    if getattr(opt, 'part_quant', False):
        quant_layer_list = list(torch.load(config.part_quant_list))
        quant_layer_list = quant_layer_list[:int(len(quant_layer_list) * opt.quant_ratio)]

    fp_layer_list: list[str] = []
    if opt.part_fp:
        with open(config.part_fp_list,'r') as f:
            lines = f.readlines()
        fp_layer_list = [line.strip() for line in lines]  # strip the '\n'
        if getattr(opt, 'fp_ratio', None) is not None:
            fp_layer_list = fp_layer_list[:int(len(fp_layer_list) * opt.fp_ratio)]
        logger.info("Set the following layers as FP: {}".format(fp_layer_list))

    with torch.no_grad():
        # --- get temp kwargs -----
        if config.model.model_type == "wan":
            # original model takes 2*bs 'y' and bs mask
            tmp_kwargs = {"mask": calib_added_kwargs["mask"][:calib_batch_size][::2].cuda()}
        else:
            tmp_kwargs = calib_added_kwargs

        qnn.set_module_name_for_quantizer(module=qnn.model)  # add the module name as attribute for each quantizer
        # _ = qnn(calib_xs[:calib_batch_size].cuda(), calib_ts[:calib_batch_size].cuda(), calib_cs[:calib_batch_size].cuda(), **tmp_kwargs)
        
        ## for w4a8 mixpricision
        # get weight_config
        if wq_params.n_bits <= 4:
            with open(opt.time_mp_config_weight, 'r') as f:
                time_mp_config_weight = yaml.safe_load(f)
            qnn.load_bitwidth_config(model=qnn, bit_config=time_mp_config_weight, bit_type='weight')
        if aq_params.n_bits <= 6:
            with open(opt.time_mp_config_act, 'r') as f:
                time_mp_config_act = yaml.safe_load(f)
            qnn.load_bitwidth_config(model=qnn, bit_config=time_mp_config_act, bit_type='act')

        # for smooth quant
        if aq_params.smooth_quant.enable:
            calib_xs_save = calib_xs
            calib_ts_save = calib_ts
            calib_cs_save = calib_cs
            calib_masks_save = calib_masks
            logger.info("begin to calculate the statistic of activation for smooth quant!")
            # assert aq_params.get('dynamic',False)
            qnn.set_smooth_quant(smooth_quant=False, smooth_quant_running_stat=True) # Now we use fp16 to save the statistic of activation
            qnn.set_quant_state(False, False)
            # The following code is the same as activation quantization, to calculate the statistic
            # Need to support different timestep, calib activation respectively, we have to split the time_steps
            calib_n_samples = config.calib_data.n_samples*2
            calib_ts = calib_ts.reshape([-1,calib_n_samples])
            # INFO: when the calib_n_samples is smaller than calib data timestep size
            # e.g., 100 / 1000, the result would be [100,10] -> [990,...x10],[980,...,x10]
            calib_n_steps = calib_ts.shape[0]
            calib_xs = calib_xs.reshape([calib_n_steps,calib_n_samples]+list(calib_xs.shape[1:])) # split the 1st dim (batch) into 2
            calib_cs = calib_cs.reshape([calib_n_steps,calib_n_samples]+list(calib_cs.shape[1:]))
            # calib_masks_shape = calib_masks.shape
            calib_masks = calib_masks.reshape([calib_n_steps,calib_n_samples]+list(calib_masks.shape[1:]))

            inds = np.arange(calib_xs.shape[1])
            np.random.shuffle(inds)
            rounds = int(calib_xs.size(1) / calib_batch_size)

            for i_ts in trange(calib_n_steps):
                assert torch.all(calib_ts[i_ts,:] == calib_ts[i_ts,0])  # ts have the same timestepe_id
                # qnn.set_timestep_for_quantizer(calib_ts[i_ts,0].item())
                for i in range(rounds):
                    if config.model.model_type == "wan":
                        _ = qnn(\
                            calib_xs[i_ts, inds[i * calib_batch_size:(i + 1) * calib_batch_size],:].cuda(),\
                            calib_ts[i_ts, inds[i * calib_batch_size:(i + 1) * calib_batch_size]].cuda(),\
                            calib_cs[i_ts, inds[i * calib_batch_size:(i + 1) * calib_batch_size],:].cuda(),\
                            mask=calib_masks[i_ts, inds[i * calib_batch_size:(i + 1) * calib_batch_size],:].cuda(),\
                        )
                    else:
                        raise NotImplementedError

            # assert aq_params.get('dynamic',False)
            qnn.set_smooth_quant(smooth_quant=True, smooth_quant_running_stat=False) # Now we use fp16 to save the statistic of activation
            qnn.set_layer_smooth_quant(model=qnn, module_name_list=fp_layer_list, smooth_quant=False, smooth_quant_running_stat=False)
            calib_xs = calib_xs_save
            calib_ts = calib_ts_save
            calib_cs = calib_cs_save
            calib_masks = calib_masks_save

        # --- the weight quantization -----
        # enable part quantization
        if getattr(opt, 'part_quant', False):
            qnn.set_layer_quant(model=qnn, module_name_list=quant_layer_list, quant_level='per_layer', weight_quant=True, act_quant=False, prefix="")
        elif opt.part_fp:
            qnn.set_quant_state(True, False)
            qnn.set_layer_quant(model=qnn, module_name_list=fp_layer_list, quant_level='per_layer', weight_quant=False, act_quant=False, prefix="")
        else:
            qnn.set_quant_state(True, False) # enable weight quantization, disable act quantization

        # For smooth quant with multiple timerange, should save many weights
        if aq_params.smooth_quant.enable:
            if aq_params.smooth_quant.get('timerange', None) is not None:
                l_range_start = [cur_timerange[0] for cur_timerange in aq_params.smooth_quant.timerange]
                for range_start in l_range_start:
                    # run calib for each
                    _ = qnn(calib_xs[:calib_batch_size].cuda(),\
                            calib_ts[:calib_batch_size].fill_(range_start).cuda(),\
                            calib_cs[:calib_batch_size].cuda(),\
                            **tmp_kwargs
                            )

            else:
                _ = qnn(calib_xs[:calib_batch_size].cuda(), calib_ts[:calib_batch_size].cuda(), calib_cs[:calib_batch_size].cuda(), **tmp_kwargs)
        else:
            _ = qnn(calib_xs[:calib_batch_size].cuda(), calib_ts[:calib_batch_size].cuda(), calib_cs[:calib_batch_size].cuda(), **tmp_kwargs)
        logger.info("weight initialization done!")
        qnn.set_quant_init_done('weight')
        torch.cuda.empty_cache()

        # --- the activation quantization -----
        # by default, use the running_mean of calibration data to determine activation quant params
        if getattr(opt, 'part_quant', False):
            qnn.set_layer_quant(model=qnn, module_name_list=quant_layer_list, quant_level='per_layer', weight_quant=True, act_quant=True, prefix="")
        elif opt.part_fp:
            qnn.set_quant_state(True, True)
            qnn.set_layer_quant(model=qnn, module_name_list=fp_layer_list, quant_level='per_layer', weight_quant=False, act_quant=False, prefix="")
        else:
            qnn.set_quant_state(True, True) # quantize activation with fixed quantized weight
        logger.info('Running stat for activation quantization')


        if aq_params.get('dynamic',False):
            logger.info('Adopting dynamic quant params, skip calculating fixed quant params')
        else:
            if not config.get('timestep_wise', False):
                # Normal activation calibration, walk through all calib data
                inds = np.arange(calib_xs.shape[0])
                # np.random.shuffle(inds)  # ERROR: using shuffle would make it mixed
                rounds = int(calib_xs.size(0) / calib_batch_size)

                for i in trange(rounds):
                    mask_ = calib_masks[inds[i * calib_batch_size:(i + 1) * calib_batch_size]][::2].cuda()
                    if config.model.model_type == "wan":
                        _ = qnn(calib_xs[inds[i * calib_batch_size:(i + 1) * calib_batch_size]].cuda(),
                            calib_ts[inds[i * calib_batch_size:(i + 1) * calib_batch_size]].cuda(),
                            calib_cs[inds[i * calib_batch_size:(i + 1) * calib_batch_size]].cuda(),
                            mask=mask_,  # INFO: original opensora model takes in 2*bs conds(y) and bs mask
                            # shape: [B, timestep, N_token], the mask are the same across timesteps, so using the 
                        )
                    else:
                        raise NotImplementedError
            else:
                # Need to support different timestep, calib activation respectively, we have to split the time_steps
                calib_n_samples = config.calib_data.n_samples*2
                calib_ts = calib_ts.reshape([-1,calib_n_samples])
                # INFO: when the calib_n_samples is smaller than calib data timestep size
                # e.g., 100 / 1000, the result would be [100,10] -> [990,...x10],[980,...,x10]
                calib_n_steps = calib_ts.shape[0]
                calib_xs = calib_xs.reshape([calib_n_steps,calib_n_samples]+list(calib_xs.shape[1:])) # split the 1st dim (batch) into 2
                calib_cs = calib_cs.reshape([calib_n_steps,calib_n_samples]+list(calib_cs.shape[1:]))
                calib_masks = calib_masks.reshape([calib_n_steps,calib_n_samples]+list(calib_masks.shape[1:]))

                inds = np.arange(calib_xs.shape[1])
                np.random.shuffle(inds)
                rounds = int(calib_xs.size(1) / calib_batch_size)

                for i_ts in trange(calib_n_steps):
                    assert torch.all(calib_ts[i_ts,:] == calib_ts[i_ts,0])  # ts have the same timestepe_id
                    qnn.set_timestep_for_quantizer(calib_ts[i_ts,0].item())
                    for i in range(rounds):
                        if config.model.model_type == "wan":
                            _ = qnn(\
                                    calib_xs[i_ts, inds[i * calib_batch_size:(i + 1) * calib_batch_size],:].cuda(),\
                                    calib_ts[i_ts, inds[i * calib_batch_size:(i + 1) * calib_batch_size]].cuda(),\
                                    calib_cs[i_ts, inds[i * calib_batch_size:(i + 1) * calib_batch_size],:].cuda(),\
                                    mask=calib_masks[i_ts, inds[i * calib_batch_size:(i + 1) * calib_batch_size],:].cuda(),\
                            )
                        else:
                            raise NotImplementedError

                # INFO: re-fill the 1000 timesteps quant params if timestep-wise 
                if config.get('timestep_wise', False):
                    qnn.repeat_timestep_wise_quant_params(calib_ts)

        qnn.set_quant_init_done('activation')
        logger.info("activation initialization done!")
        torch.cuda.empty_cache()

    # ----------------------- get the quant params (training opt), using the calibration data -------------------------------------
    # import ipdb; ipdb.set_trace()
    weight_optimization = False
    if config.quant.weight.optimization is not None:
        if config.quant.weight.optimization.params is not None:
            weight_optimization = True
    act_optimization = False
    if config.quant.activation.optimization is not None:
        if config.quant.activation.optimization.params is not None:
            act_optimization = True
    use_optimization = any([weight_optimization, act_optimization])

    qnn.fp_layer_list = fp_layer_list
    
    if not use_optimization:  # no need for optimization-based quantization
        pass
    else:
        # INFO: get the quant parameters
        qnn.train()  # setup the train_mode
        opt_d = {}
        if weight_optimization:
            opt_d['weight'] = getattr(config.quant,'weight').optimization.params.keys()
        else:
            opt_d['weight'] = None
        if act_optimization:
            opt_d['activation'] = getattr(config.quant,'activation').optimization.params.keys()
        else:
            opt_d['activation'] = None
        qnn.replace_quant_buffer_with_parameter(opt_d)


        # --- the weight quantization (with optimization) -----
        if not weight_optimization:
            logger.info("No quant parmas, skip optimizing weight quant parameters")
        else:
            qnn.set_quant_state(True, False)  # use FP activation
            opt_target = 'weight'
            # --- unpack the config ----
            param_types = list(config.quant.weight.optimization.params.keys())
            if 'alpha' in param_types:
                assert config.quant.weight.quantizer.round_mode == 'learned_hard_sigmoid'  # check adaround stat
            our_model_reconstruction(qnn,qnn,calib_data,config,param_types,opt_target)  # DEBUG_ONLY
            logger.info("Finished optimizing param {} for layer's {}, saving temporary checkpoint...".format(param_types, opt_target))
            torch.save(qnn.get_quant_params_dict(), os.path.join(outpath, "ckpt.pth"))

        # --- the activation quantization (with optimization) -----
        if not act_optimization:
            logger.info("No quant parmas, skip optimizing activation quant parameters")
        else:
            pass

        qnn.replace_quant_parameter_with_buffers(opt_d)  # replace back to buffer for saving



    # save the quant params
    logger.info("Saving calibrated quantized DiT model")
    quant_params_dict = qnn.get_quant_params_dict()
    # import ipdb; ipdb.set_trace()
    torch.save(quant_params_dict, os.path.join(outpath, "ckpt.pth"))

if __name__ == "__main__":
    main()

# FreqFiT - Frequency Fine-Tuning: How Transformers Learned to Stop Worrying and Love Bandwidth

This repository contains the official PyTorch implementation for `FreqFiT - ECCV24 submission #10513`.

<img src="https://github.com/tsly123/FreqFiT/blob/main/assets/freqfit.png">

This repository is heavily based on the official PyTorch implementation of [Visual Prompt Tuning](https://github.com/KMnP/vpt) (ECCV22)

## Environment settings

See `env_setup.sh` or `assets/freqfit.yml`

## Experiments

### Datasets preparation

Please follow the [VPT Datasets preperation](https://github.com/KMnP/vpt?tab=readme-ov-file#datasets-preperation) and `VTAB_SETUP.md`

### Pre-trained model preparation

Download and place the pre-trained Transformer-based backbones to the `pretrained` folder or to
`MODEL.MODEL_ROOT`. 

Note that, for MoCo v3, different from VPT, we use the self-supervised pre-trained weights.

Once downloaded, modify the pre-trained backbones names `MODEL_ZOO` in `src/build_vit_backbone.py` accordingly.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Pre-trained Backbone</th>
<th valign="bottom">Pre-trained Objective</th>
<th valign="bottom">Link</th>
<!-- TABLE BODY -->
<tr><td align="left">ViT-B/16</td>
<td align="center">Supervised</td>
<td align="center"><a href="https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz">link</a></td>
</tr>
<tr><td align="left">ViT-B/16</td>
<td align="center">MoCo v3</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar">link</a></td>
</tr>
<tr><td align="left">ViT-B/16</td>
<td align="center">MAE</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth">link</a></td>
</tr>
<tr><td align="left">ViT-B/16</td>
<td align="center">CLIP</td>
<td align="center"><a href="https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt">link</a></td>
</tr>
</tbody></table>

### Run experiments
Modify the `run.sh` as your reference. Then run:

```
bash run.sh [data_name] [encoder] [batch_size] [base_lr] [wd_lr] [num_tokens] [adapter_ratio]
```
Note that, `num_tokens` and `adapter_ratio` are for VPT and Adapter, respectively. For example, for the `DTD` dataset on `CLIP` with `Linear` tuning, execute:
```
bash run.sh dtd clip_vitb16 128 0.1 0.01
```

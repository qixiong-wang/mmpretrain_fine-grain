import copy
import math
from functools import partial
from typing import Optional, List, Tuple
import einops
import torch
from mmcv.cnn.bricks.transformer import build_transformer_layer
from mmengine import digit_version
from mmengine.dist import is_main_process
from mmengine.hooks import Hook
from mmengine.model import BaseModule
# from peft import LoraConfig, get_peft_model
from torch import nn

from mmpretrain.models import ImageClassifier, LinearClsHead
from mmpretrain.models.utils import  ClsDataPreprocessor
from mmpretrain.registry import MODELS, HOOKS
from mmpretrain.structures import DataSample
import torch.nn.functional as F
import pickle
from .clip import CLIP
from transformers import AutoTokenizer
from transformers import ChineseCLIPModel, ChineseCLIPConfig
from mmpretrain.registry import MODELS, TOKENIZER

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


@MODELS.register_module()
class TransformerFusionNeck(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_modality=8,
                 with_cls_token=True,
                 num_encoder_layers=3,
                 drop_img_rate=0,
                 drop_token_rate=0,
                 drop_extra_rate=0,
                 drop_modality_rate=0,
                 init_cfg=None,
                 ):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.with_cls_token = with_cls_token
        self.drop_img_rate = drop_img_rate
        self.drop_token_rate = drop_token_rate
        self.drop_extra_rate = drop_extra_rate
        self.drop_modality_rate = drop_modality_rate

        if self.with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, self.embed_dims))
        if num_modality > 0:
            self.modality_pe = nn.Parameter(torch.zeros(num_modality, self.embed_dims))

        mlp_ratio = 4
        transformer_layer = dict(
            type='BaseTransformerLayer',
            attn_cfgs=[
                dict(
                    type='MultiheadAttention',
                    embed_dims=embed_dims,
                    num_heads=8,
                    attn_drop=0.1,
                    proj_drop=0.1,
                    # dropout_layer=dict(type='Dropout', drop_prob=0.1)
                ),
            ],
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=embed_dims,
                feedforward_channels=embed_dims * mlp_ratio,
                num_fcs=2,
                act_cfg=dict(type='GELU'),
                ffn_drop=0.1,
                add_identity=True),
            operation_order=('norm', 'self_attn', 'norm', 'ffn'),
            norm_cfg=dict(type='LN'),
            batch_first=True
        )

        self.layers = nn.ModuleList()
        transformer_layers = [
            copy.deepcopy(transformer_layer) for _ in range(num_encoder_layers)
        ]
        for i in range(num_encoder_layers):
            self.layers.append(build_transformer_layer(transformer_layers[i]))
        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].pre_norm

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat_dict):
        # feat_dict is a dict of features
        # add modality pe
        if hasattr(self, 'modality_pe'):
            pe_idx = 0
            for k, v in feat_dict.items():
                if '_attn_masks' not in k:
                    pe = self.modality_pe[pe_idx:pe_idx+1, :]
                    feat_dict[k] = v + pe
                    pe_idx += 1

        main_img_feat = feat_dict['img']  # B N C D
        # assert len(main_img_feat.shape) == 4
        main_string_feat = feat_dict['input_main_string']  # B C D

        # get attn masks
        img_attn_masks = feat_dict['img_attn_masks'].bool()
        text_attn_masks = feat_dict['input_main_string_attn_masks'].bool()

        # extra feat
        extra_feat_keys = [k for k in feat_dict.keys() if 'input_' in k and 'input_main_string' not in k and '_attn_masks' not in k]
        extra_feat = []
        extra_attn_masks = []
        for key in extra_feat_keys:
            extra_feat.append(feat_dict[key])
            extra_attn_masks.append(feat_dict[key+'_attn_masks'].bool())

        # concat extra feat
        if len(extra_feat) > 0:
            extra_feat = torch.cat(extra_feat, dim=1)
            extra_attn_masks = torch.cat(extra_attn_masks, dim=1).bool()

        # dropout when training
        if self.training:
            # image: B N C D, dropout on N
            drop_img_mask = torch.rand(main_img_feat.shape[0], main_img_feat.shape[1], device=main_img_feat.device) < self.drop_img_rate
            # import pdb; pdb.set_trace()
            # no_img_mask = drop_img_mask | (~img_attn_masks)
            # drop_img_mask[no_img_mask.sum(dim=-1) == main_img_feat.shape[1]] = False
            # img_attn_masks = img_attn_masks * (~drop_img_mask)

            # # concat img
            img_feat = main_img_feat
            # per_img_feat_dim = main_img_feat.shape[2]
            # img_feat = einops.rearrange(main_img_feat, 'b n c d -> b (n c) d')
            # img_attn_masks = einops.repeat(img_attn_masks, 'b n -> b (n c)', c=per_img_feat_dim)

            # randomly drop on tokens
            # drop_img_mask = torch.rand(img_attn_masks.shape[0], img_attn_masks.shape[1], device=img_feat.device) < self.drop_token_rate
            drop_text_mask = torch.rand(main_string_feat.shape[0], main_string_feat.shape[1], device=main_string_feat.device) < self.drop_token_rate
            # img_attn_masks = img_attn_masks * (~drop_img_mask)
            img_attn_masks = img_attn_masks.unsqueeze(1).expand_as(drop_img_mask)

            text_attn_masks = text_attn_masks * (~drop_text_mask)

            # randomly drop on modalities
            drop_modality_mask = torch.rand(img_feat.shape[0], device=img_feat.device) < self.drop_modality_rate
            # select which modality to drop
            drop_img_modality = torch.rand(img_feat.shape[0], device=img_feat.device) < 0.5
            drop_img_mask = drop_modality_mask & drop_img_modality
            drop_text_mask = drop_modality_mask & (~drop_img_modality)
            img_attn_masks = img_attn_masks * (~drop_img_mask)[..., None]
            text_attn_masks = text_attn_masks * (~drop_text_mask)[..., None]

            if len(extra_feat) > 0:
                # randomly drop on extra
                drop_extra_mask = torch.rand(extra_feat.shape[0], device=extra_feat.device) < self.drop_extra_rate
                extra_attn_masks = extra_attn_masks * (~drop_extra_mask)[..., None]

                # randomly drop on tokens
                drop_extra_mask = torch.rand(extra_attn_masks.shape[0], extra_attn_masks.shape[1], device=extra_feat.device) < self.drop_token_rate
                extra_attn_masks = extra_attn_masks * (~drop_extra_mask)
        else:
            # concat img

            img_feat = main_img_feat
            # per_img_feat_dim = main_img_feat.shape[2]
            # img_feat = einops.rearrange(main_img_feat, 'b n c d -> b (n c) d')
            # img_attn_masks = einops.repeat(img_attn_masks, 'b n -> b (n c)', c=per_img_feat_dim)

        # concat cls_token, img and text
        cls_tokens = self.cls_token.expand(img_feat.shape[0], -1, -1)
        if len(extra_feat) > 0:
            x = torch.cat((cls_tokens, img_feat, main_string_feat, extra_feat), dim=1)
            cls_tokens_attn_masks = torch.ones(cls_tokens.shape[:2]).bool().to(img_feat.device)
            attention_mask = torch.cat((cls_tokens_attn_masks, img_attn_masks, text_attn_masks, extra_attn_masks), dim=1)
        else:
            x = torch.cat((cls_tokens, img_feat, main_string_feat), dim=1)
            cls_tokens_attn_masks = torch.ones(cls_tokens.shape[:2]).bool().to(img_feat.device)
            attention_mask = torch.cat((cls_tokens_attn_masks, img_attn_masks, text_attn_masks), dim=1)

        attention_mask = ~attention_mask.bool()
        # For a binary mask, a ``True`` value indicates that the
        #             corresponding position is not allowed to attend.
        for layer in self.layers:
            x = layer(x, query_key_padding_mask=attention_mask, key_padding_mask=attention_mask)
        return x, x[:, 0, :]



@MODELS.register_module()
class CLIPTransformerFusionClassifier(ImageClassifier):
    def __init__(
            self,
            text_backbone,
            vocab_size,
            transformer_width,
            proj_dim,
            context_length,
            tokenizer,
            text_keys=['input_main_string'],
            freeze_backbone=False,
            vision_project=None,
            text_project=None,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.text_keys = text_keys
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.tokenizer = TOKENIZER.build(tokenizer)
        text_backbone['attn_mask'] = self.build_attention_mask()

        # self.text_backbone = MODELS.build(text_backbone)
        self.transformer = MODELS.build(text_backbone)
        self.ln_final = LayerNorm(transformer_width)

        if vision_project is not None:
            self.vision_projector = MODELS.build(vision_project)
        # if text_project is not None:
        #     self.text_projector = MODELS.build(text_project)

        if is_main_process():
            self.print_trainable_parameters()

        self.text_projection = nn.Parameter(
            torch.empty(transformer_width, proj_dim))
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width))
        
    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        trainable_names = []
        for name, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
                trainable_names.append(name)
        print(f"Trainable parameter names: {trainable_names}")
        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )

    def tokenize(self, texts):
        """Returns the tokenized representation of given input string(s)

        Args:
            texts (Union[str, List[str]]): An input string or a list of input
                strings to tokenize
            context_length (int): The context length to use. Defaults to 52.

        Returns:
            torch.Tensor: Resulting tokens.
        """
        if isinstance(texts, str):
            texts = [texts]

        all_tokens = []
        for text in texts:
            # adapt the text to Chinese BERT vocab
            # text = text.lower().replace('“', "\"").replace('”', "\"")

            # add special tokens
            all_tokens.append(
                self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(text))[:self.context_length - 2])

        result = torch.zeros(
            len(all_tokens), self.context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            assert len(tokens) <= self.context_length
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result
    
    def extract_text_feat(self, texts: torch.Tensor) -> torch.Tensor:
        """The function to extract text latent features."""
        x = self.token_embedding(texts)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND

        x = self.transformer(x)[0]

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding
        # (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]),
              texts.argmax(dim=-1)] @ self.text_projection
        return x

    def build_attention_mask(self):
        # lazily create causal attention mask,
        # with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float('-inf'))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    
    def extract_feat(
            self,
            inputs: torch.Tensor,
            data_samples: Optional[List[DataSample]] = None
    ):

        batch_size = inputs.shape[0]
        feat_dict = dict()
        # img_attn_masks = torch.ones(inputs.shape[0],device=inputs.device)

        # for img feat
        # imgs = einops.rearrange(inputs, 'b n c h w -> (b n) c h w')
        img_feat = self.backbone(inputs)

        img_feat = self.vision_projector(img_feat)[0]
        # feat_dict['img'] = img_feat
        # feat_dict['img_attn_masks'] = img_attn_masks
        # for extra discrete and continuous feat
        for text_key in self.text_keys:

            # input_ids = torch.cat([x.get(text_key)['input_ids'] for x in data_samples]).to(inputs.device)
            # attention_mask = torch.cat([x.get(text_key)['attention_mask'] for x in data_samples]).to(inputs.device)

            # # token_type_ids = torch.cat([x.get(text_key)['token_type_ids'] for x in data_samples]).to(inputs.device)
            # feat_dict[text_key+'_attn_masks'] = attention_mask

            # text_input_dict = dict(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     # token_type_ids=token_type_ids,
            # )
            # text_feat = self.text_backbone(input_ids)  ## 16, 512, 768

            texts = [x.get(text_key) for x in data_samples]

            tokenized_texts = self.tokenize(texts).to(inputs.device)
            text_feat = self.extract_text_feat(tokenized_texts)
            # hidden_states = text_feat.last_hidden_state
            # hidden_states = self.text_projector(hidden_states)

            feat_dict[text_key] = text_feat
            # if self.dense_text:
            #     pred_feats = hidden_states
            # hidden_states = self.text_projector(hidden_states)

            # index_range = torch.arange(attention_mask.shape[-1]).to(inputs.device)
            # index_range = index_range.expand(batch_size, -1)
            # index_range = index_range.masked_fill(attention_mask.bool(), attention_mask.shape[-1])
            # min_index = torch.argmin(index_range, dim=-1)
            # min_index[min_index == 0] = attention_mask.shape[-1] - 1
            # pred_feats = hidden_states[torch.arange(batch_size), min_index]
            # pred_feats = pred_feats.unsqueeze(1)
            # pred_feats = pred_feats.to(inputs.dtype)
            # feat_dict[text_key] = pred_feats
        # feats = self.neck(feat_dict)
        feats = torch.cat((img_feat, text_feat), dim=1)
        
        return feats, feats

    def loss(self, inputs: torch.Tensor,
             data_samples: List[DataSample]) -> dict:
        feats = self.extract_feat(inputs, data_samples)
        losses = self.head.loss(feats, data_samples)
        return losses

    def predict(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                **kwargs) -> List[DataSample]:
        feats = self.extract_feat(inputs, data_samples)
        res = self.head.predict(feats, data_samples, **kwargs)
        return res
    
# @MODELS.register_module()
# class MultiImgClsDataPreprocessor(ClsDataPreprocessor):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def forward(self, data: dict, training: bool = False) -> dict:

#         inputs = data['inputs']
#         bs, n = inputs.shape[:2]
#         inputs = einops.rearrange(inputs, 'b n c h w -> (b n) c h w')
#         data['inputs'] = inputs
#         data = super().forward(data, training)
#         data['inputs'] = einops.rearrange(data['inputs'], '(b n) c h w -> b n c h w', b=bs)
#         return data



@MODELS.register_module()
class MultiVersionLinearClsHead(LinearClsHead):
    def __init__(
        self,
        data_versions=[2, 3],
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.data_versions = data_versions
        self.fc = nn.ModuleList()
        for _ in range(len(data_versions)):
            self.fc.append(nn.Sequential(
                nn.Linear(self.in_channels, self.in_channels // 2),
                nn.ReLU(inplace=True),
                nn.Linear(self.in_channels // 2, self.num_classes),
            ))

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        pass

    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> dict:
        pre_logits = self.pre_logits(feats)
        data_versions = torch.tensor([x.data_version for x in data_samples]).to(pre_logits.device)
        losses = dict()
        for i, data_version in enumerate(self.data_versions):
            version_mask = data_versions == data_version
            if version_mask.sum() > 0:
                cls_score = self.fc[i](pre_logits[version_mask])
                data_samples_tmp = [x for x in data_samples if x.data_version == data_version]
                losses_tmp = self._get_loss(cls_score, data_samples_tmp)

            else:
                cls_score = self.fc[i](pre_logits[:1])
                losses_tmp = dict(loss=cls_score.sum() * 0)
                if self.cal_acc:
                    losses_tmp['accuracy_top-1'] = pre_logits.sum() * 0
            # add prefix
            losses_tmp = {f'{k}_{data_version}': v for k, v in losses_tmp.items()}
            losses.update(losses_tmp)

        return losses

    def predict(
        self,
        feats: Tuple[torch.Tensor],
        data_samples: Optional[List[Optional[DataSample]]] = None
    ) -> List[DataSample]:
        pre_logits = self.pre_logits(feats)
        data_versions = torch.tensor([x.data_version for x in data_samples]).to(pre_logits.device)
        results = []
        for i, data_version in enumerate(self.data_versions):
            version_mask = data_versions == data_version
            if version_mask.sum() > 0:
                cls_score = self.fc[i](pre_logits[version_mask])
                data_samples_tmp = [x for x in data_samples if x.data_version == data_version]
                results_tmp = self._get_predictions(cls_score, data_samples_tmp)
                results.extend(results_tmp)
        return results

@MODELS.register_module()
class HFCNCLIPVision(BaseModule):
    def __init__(
            self,
            from_pretrained=True,
            config_dir='OFA-Sys/chinese-clip-vit-base-patch16',
            freeze=False,
            lora_cfg=None,
            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        if from_pretrained:
            model = ChineseCLIPModel.from_pretrained(config_dir)
        else:
            model = ChineseCLIPModel(config=ChineseCLIPConfig.from_pretrained(config_dir))
        vision_model = model.vision_model

        if freeze:
            for param in vision_model.parameters():
                param.requires_grad = False

        # if lora_cfg is not None:
        #     default_lora_cfg = dict(
        #         r=16,
        #         lora_alpha=16,
        #         target_modules=["query", "value"],
        #         lora_dropout=0.0,
        #     )
        #     default_lora_cfg.update(lora_cfg)
        #     config = LoraConfig(**default_lora_cfg)
        #     self.vision_model = get_peft_model(vision_model, config)
        #     if is_main_process():
        #         self.vision_model.print_trainable_parameters()
        # else:
        self.vision_model = vision_model

    def init_weights(self):
        pass

    def merge_lora_weights(self):
        self.vision_model.merge_and_unload()

    def forward(self, *args, **kwargs):
        return self.vision_model(*args, **kwargs)


@MODELS.register_module()
class HFCNCLIPText(BaseModule):
    def __init__(
            self,
            from_pretrained=True,
            config_dir='OFA-Sys/chinese-clip-vit-base-patch16',
            lora_cfg=None,
            freeze=False,
            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        if from_pretrained:
            model = ChineseCLIPModel.from_pretrained(config_dir)
        else:
            model = ChineseCLIPModel(config=ChineseCLIPConfig.from_pretrained(config_dir))

        text_model = model.text_model

        if freeze:
            for param in text_model.parameters():
                param.requires_grad = False

        # if lora_cfg is not None:
        #     default_lora_cfg = dict(
        #         r=16,
        #         lora_alpha=16,
        #         target_modules=["query", "value"],
        #         lora_dropout=0.0,
        #     )
        #     default_lora_cfg.update(lora_cfg)
        #     config = LoraConfig(**default_lora_cfg)
        #     self.text_model = get_peft_model(text_model, config)
        #     if is_main_process():
        #         self.text_model.print_trainable_parameters()
        # else:
        self.text_model = text_model

    def init_weights(self):
        pass

    def merge_lora_weights(self):
        self.text_model.merge_and_unload()

    def forward(self, *args, **kwargs):
        return self.text_model(*args, **kwargs)
import logging
import os
import pickle
from collections import OrderedDict

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter

import common
import metrics
from utils import plot_segmentation_images,plot_overlay_segmentation

LOGGER = logging.getLogger(__name__)

def init_weight(m):
    """Initializes weights for linear and convolutional layers using Xavier normal initialization."""
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)

class FourierTransformModule(torch.nn.Module):
    """Extracts frequency domain features using Fast Fourier Transform (FFT)."""
    def __init__(self, feature_dim, frequency_dim=None, use_phase=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.frequency_dim = frequency_dim if frequency_dim is not None else feature_dim // 2
        self.use_phase = use_phase
        
        # Linear layers for processing frequency domain features
        input_freq_dim = self.frequency_dim * 2 if use_phase else self.frequency_dim
        self.frequency_linear = nn.Sequential(
            nn.Linear(input_freq_dim, self.frequency_dim),
            nn.ReLU(),
            nn.Linear(self.frequency_dim, self.frequency_dim),
            nn.BatchNorm1d(self.frequency_dim)
        )
        
        # Learnable standard deviation for frequency noise regularization
        self.freq_noise_std = nn.Parameter(torch.tensor(0.02))
        
    def forward(self, x, add_noise=True):
        # Perform Fast Fourier Transform along the last dimension
        x_fft = torch.fft.fft(x, dim=-1)
        
        # Extract magnitude and phase components from the first half of the spectrum
        magnitude = torch.abs(x_fft)[:, :self.frequency_dim]
        
        if self.use_phase:
            phase = torch.angle(x_fft)[:, :self.frequency_dim]
            freq_features = torch.cat([magnitude, phase], dim=-1)
        else:
            freq_features = magnitude
            
        # Process frequency features through linear layers
        freq_output = self.frequency_linear(freq_features)
        
        # Add Gaussian noise during training for regularization
        if add_noise and self.training:
            freq_noise = torch.normal(0, self.freq_noise_std, freq_output.shape).to(freq_output.device)
            freq_output = freq_output + freq_noise
            
        return freq_output

class SpatialAttentionModule(torch.nn.Module):
    """Applies spatial attention to enhance feature importance based on channel and feature weights."""
    def __init__(self, feature_dim, reduction_ratio=16):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Channel attention branch using Adaptive Average Pooling and Conv1d
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(feature_dim, feature_dim // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv1d(feature_dim // reduction_ratio, feature_dim, 1),
            nn.Sigmoid()
        )
        
        # Feature importance weighting branch using linear layers
        self.feature_weight = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // reduction_ratio),
            nn.ReLU(),
            nn.Linear(feature_dim // reduction_ratio, feature_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (batch_size, feature_dim)
        
        # Apply channel attention by expanding features to (B, C, 1)
        x_expanded = x.unsqueeze(-1)
        channel_att = self.channel_attention(x_expanded).squeeze(-1)
        
        # Apply feature importance weighting
        feature_att = self.feature_weight(x)
        
        # Combine attention weights from both branches
        attention = channel_att * feature_att
        
        # Apply attention to input features
        return x * attention

class AdvancedFeatureFusion(torch.nn.Module):
    """Fuses spatial and frequency features using various strategies."""
    def __init__(self, spatial_dim, frequency_dim, output_dim, fusion_type='attention'):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.frequency_dim = frequency_dim
        self.output_dim = output_dim
        self.fusion_type = fusion_type
        
        if fusion_type == 'attention':
            # Attention-based fusion layers
            self.spatial_proj = nn.Linear(spatial_dim, output_dim)
            self.frequency_proj = nn.Linear(frequency_dim, output_dim)
            self.attention = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, 2),
                nn.Softmax(dim=1)
            )
        elif fusion_type == 'gated':
            # Gated fusion layers
            self.spatial_proj = nn.Linear(spatial_dim, output_dim)
            self.frequency_proj = nn.Linear(frequency_dim, output_dim)
            self.gate = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.Sigmoid()
            )
        else:  # 'concat'
            # Simple concatenation and projection
            self.fusion_proj = nn.Linear(spatial_dim + frequency_dim, output_dim)
            
    def forward(self, spatial_features, frequency_features):
        if self.fusion_type == 'attention':
            spatial_proj = self.spatial_proj(spatial_features)
            frequency_proj = self.frequency_proj(frequency_features)
            
            # Compute attention weights for fusion
            combined = torch.cat([spatial_proj, frequency_proj], dim=1)
            attention_weights = self.attention(combined)
            
            # Weighted sum of projected features
            output = attention_weights[:, 0:1] * spatial_proj + attention_weights[:, 1:2] * frequency_proj
            
        elif self.fusion_type == 'gated':
            spatial_proj = self.spatial_proj(spatial_features)
            frequency_proj = self.frequency_proj(frequency_features)
            
            # Apply gating mechanism
            combined = torch.cat([spatial_proj, frequency_proj], dim=1)
            gate = self.gate(combined)
            
            output = gate * spatial_proj + (1 - gate) * frequency_proj
            
        else:  # concat
            # Simple concatenation and projection
            combined = torch.cat([spatial_features, frequency_features], dim=1)
            output = self.fusion_proj(combined)
            
        return output

class EnhancedDiscriminator(torch.nn.Module):
    """Discriminator with adaptive margin and regularization options."""
    def __init__(self, in_planes, n_layers=2, hidden=None, dropout=0.1, 
                 initial_margin=0.5, learnable_margin=True, use_spectral_norm=False):
        super().__init__()

        _hidden_current = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        
        # Build discriminator body with linear layers, BatchNorm, LeakyReLU, and Dropout
        for i in range(n_layers-1):
            _in = in_planes if i == 0 else _hidden_current
            _hidden_current = int(_hidden_current // 1.5) if hidden is None else hidden
            
            linear_layer = torch.nn.Linear(_in, _hidden_current)
            if use_spectral_norm:
                linear_layer = torch.nn.utils.spectral_norm(linear_layer)
                
            self.body.add_module(f'block{i+1}',
                torch.nn.Sequential(
                    linear_layer,
                    torch.nn.BatchNorm1d(_hidden_current),
                    torch.nn.LeakyReLU(0.2),
                    torch.nn.Dropout(dropout)
                ))
        
        final_head_input_dim = _hidden_current if n_layers > 1 else in_planes
        
        # Output head for anomaly score
        final_linear = torch.nn.Linear(final_head_input_dim, 1, bias=False)
        if use_spectral_norm:
            final_linear = torch.nn.utils.spectral_norm(final_linear)
        self.head = final_linear
        
        # Learnable adaptive margin for boundary
        self.learnable_margin = learnable_margin
        if self.learnable_margin:
            self.adaptive_margin = nn.Parameter(torch.tensor(initial_margin, dtype=torch.float32))
        else:
            self.register_buffer('adaptive_margin', torch.tensor(initial_margin, dtype=torch.float32))

        self.apply(init_weight)

    def forward(self, x):
        x = self.body(x)
        score = self.head(x)
        return score, self.adaptive_margin

class SimplifiedMoE(torch.nn.Module):
    """Simplified Mixture-of-Experts module for anomaly detection."""
    def __init__(self, input_dim, output_dim, num_experts=4, top_k=2, 
                 temperature=1.0, expert_dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature
        
        # Expert networks (simple linear layers)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(expert_dropout),
                nn.Linear(output_dim, output_dim)
            ) for _ in range(num_experts)
        ])
        
        # Router network to select experts
        self.router = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Compute routing weights
        router_weights = self.router(x) / self.temperature
        
        # Select top-k experts based on weights
        top_k_weights, top_k_indices = torch.topk(router_weights, self.top_k, dim=1)
        top_k_weights = F.softmax(top_k_weights, dim=1)
        
        # Calculate expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1) # (batch_size, num_experts, output_dim)
        
        # Aggregate expert outputs based on selected experts and their weights
        output = torch.zeros(batch_size, expert_outputs.shape[-1], device=x.device)
        for i in range(batch_size):
            selected_outputs = expert_outputs[i, top_k_indices[i]]
            weighted_output = (top_k_weights[i].unsqueeze(-1) * selected_outputs).sum(dim=0)
            output[i] = weighted_output
            
        # Returns output and a zero auxiliary loss for interface compatibility
        return output, torch.tensor(0.0, device=x.device)

class AdvancedProjection(torch.nn.Module):
    """Projection layer integrating spatial and frequency features with optional attention and MoE."""
    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0, 
                 use_fourier=True, use_attention=True, use_moe=True, 
                 fourier_dim=None, fusion_type='attention',moe_temperature=1.0,
                 expert_dropout=0.1,reduction_ratio=16,num_experts=4, top_k=2,
                 use_fourier_phase=True):
        super().__init__()
        
        if out_planes is None:
            out_planes = in_planes
            
        self.use_fourier = use_fourier
        self.use_attention = use_attention
        self.use_moe = use_moe
        self.moe_temperature = moe_temperature
        self.expert_dropout = expert_dropout
        self.reduction_ratio = reduction_ratio
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_fourier_phase = use_fourier_phase
        
        # Spatial feature processing layers
        self.spatial_layers = torch.nn.Sequential()
        _in = in_planes
        _out = out_planes
        for i in range(n_layers):
            if i > 0:
                _in = _out
            self.spatial_layers.add_module(f"{i}fc", torch.nn.Linear(_in, _out))
            if i < n_layers - 1 and layer_type > 1:
                self.spatial_layers.add_module(f"{i}relu", torch.nn.LeakyReLU(.2))
                
        # Fourier Transform module for frequency features
        if self.use_fourier:
            self.fourier_module = FourierTransformModule(
                feature_dim=in_planes, 
                frequency_dim=fourier_dim or in_planes // 2,
                use_phase=use_fourier_phase
            )
            
            # Feature fusion module to combine spatial and frequency features
            freq_output_dim = fourier_dim or in_planes // 2
            self.feature_fusion = AdvancedFeatureFusion(
                spatial_dim=out_planes,
                frequency_dim=freq_output_dim,
                output_dim=out_planes,
                fusion_type=fusion_type
            )
        
        # Spatial Attention module
        if self.use_attention:
            self.attention_module = SpatialAttentionModule(out_planes,
                                                           reduction_ratio = self.reduction_ratio)
            
        # Simplified MoE (optional)
        if self.use_moe:
            self.moe_module = SimplifiedMoE(out_planes, out_planes,temperature=self.moe_temperature,
                                            num_experts=self.num_experts, top_k=self.num_experts, 
                                            expert_dropout=self.expert_dropout)
            
        self.apply(init_weight)
    
    def forward(self, x, add_noise=True):
        # Process spatial features
        spatial_features = self.spatial_layers(x)
        
        if self.use_fourier:
            # Extract frequency features
            frequency_features = self.fourier_module(x, add_noise=add_noise)
            
            # Fuse spatial and frequency features
            fused_features = self.feature_fusion(spatial_features, frequency_features)
        else:
            fused_features = spatial_features
            
        # Apply spatial attention
        if self.use_attention:
            attended_features = self.attention_module(fused_features)
        else:
            attended_features = fused_features
            
        # Apply MoE (optional)
        if self.use_moe:
            moe_output, aux_loss = self.moe_module(attended_features)
            return moe_output, aux_loss
        else:
            return attended_features, torch.tensor(0.0, device=x.device)

class TBWrapper:
    """Wrapper for TensorBoard SummaryWriter."""
    def __init__(self, log_dir):
        self.g_iter = 0
        self.logger = SummaryWriter(log_dir=log_dir)
    
    def step(self):
        """Increments global iteration counter."""
        self.g_iter += 1

class MAADNet(torch.nn.Module):
    """Multi-Aspect Anomaly Detector (MAADNet) for industrial fault diagnosis."""
    def __init__(self, device):
        super().__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        meta_epochs=1,
        aed_meta_epochs=1,
        gan_epochs=1,
        noise_std=0.015,
        mix_noise=3,
        dsc_layers=2,
        dsc_hidden=None,
        dsc_margin=.5,
        dsc_lr=0.0002,
        train_backbone=False,
        auto_noise=0,
        cos_lr=False,
        lr=1e-3,
        pre_proj=1,
        proj_layer_type=0,
        use_enhanced_projection=True,
        use_fourier=True,
        use_attention=True,
        use_simplified_moe=True,
        fourier_dim=None,
        fusion_type='attention',
        dsc_dropout=0.1,
        dsc_spectral_norm=True,
        dsc_learnable_margin=True,
        fourier_use_phase=True,
        attention_reduction_ratio=16,
        moe_num_experts=4,
        moe_top_k=2,
        moe_temperature=1.0,
        moe_expert_dropout=0.1,
        **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        # Feature extraction from backbone
        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device, train_backbone
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        # Preprocessing of extracted features
        preprocessing = common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        # Aggregation of preprocessed features
        preadapt_aggregator = common.Aggregator(target_dim=target_embed_dimension)
        _ = preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        # Anomaly segmentation module
        self.anomaly_segmentor = common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.meta_epochs = meta_epochs
        self.lr = lr
        self.cos_lr = cos_lr
        self.train_backbone = train_backbone
        if self.train_backbone:
            self.backbone_opt = torch.optim.AdamW(
                self.forward_modules["feature_aggregator"].backbone.parameters(), lr
            )
        
        self.aed_meta_epochs = aed_meta_epochs
        self.pre_proj = pre_proj
        self.use_enhanced_projection = use_enhanced_projection

        # Advanced Projection Layer (with Fourier, Attention, MoE)
        if self.use_enhanced_projection and self.pre_proj > 0:
            self.advanced_projection = AdvancedProjection(
                in_planes=self.target_embed_dimension,
                out_planes=self.target_embed_dimension,
                n_layers=pre_proj,
                layer_type=proj_layer_type,
                use_fourier=use_fourier,
                use_attention=use_attention,
                use_moe=use_simplified_moe,
                fourier_dim=fourier_dim,
                fusion_type=fusion_type,
                moe_temperature=moe_temperature,
                expert_dropout=moe_expert_dropout,
                reduction_ratio=attention_reduction_ratio,
                num_experts=moe_num_experts,
                top_k=moe_top_k,
                fourier_use_phase=fourier_use_phase
            )
            self.advanced_projection.to(self.device)
            self.proj_opt = torch.optim.AdamW(self.advanced_projection.parameters(), lr * 0.1)
        elif self.pre_proj > 0:
            # Fallback to original Projection layer
            self.pre_projection = Projection(
                self.target_embed_dimension, self.target_embed_dimension, 
                pre_proj, proj_layer_type
            )
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.AdamW(self.pre_projection.parameters(), lr * 0.1)
        else:
            self.advanced_projection = None
            self.pre_projection = None
            self.proj_opt = None

        # Enhanced Discriminator
        self.auto_noise = [auto_noise, None]
        self.dsc_lr = dsc_lr
        self.gan_epochs = gan_epochs
        self.mix_noise = mix_noise
        self.noise_std = noise_std
        
        self.discriminator = EnhancedDiscriminator(
            in_planes=self.target_embed_dimension,
            n_layers=dsc_layers,
            hidden=dsc_hidden,
            dropout=dsc_dropout,
            initial_margin=dsc_margin,
            learnable_margin=dsc_learnable_margin,
            use_spectral_norm=dsc_spectral_norm
        )
        self.discriminator.to(self.device)
        self.dsc_opt = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.dsc_lr, weight_decay=1e-5
        )
        self.dsc_schl = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.dsc_opt, (meta_epochs - aed_meta_epochs) * gan_epochs, self.dsc_lr * 0.4
        )

        self.model_dir = ""
        self.dataset_name = ""
        self.logger = None

    def set_model_dir(self, model_dir, dataset_name):
        """Sets directories for model checkpoints and TensorBoard logs."""
        self.model_dir = model_dir 
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.ckpt_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.logger = TBWrapper(self.tb_dir)

    def embed(self, data):
        """Generates feature embeddings for input data."""
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            aux_losses = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                    input_image = image.to(torch.float).to(self.device)
                with torch.no_grad():
                    feats, _, aux_loss = self._embed(input_image)
                    features.append(feats)
                    if aux_loss is not None:
                        aux_losses.append(aux_loss)
            return features, aux_losses
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False, evaluation=False):
        """Returns feature embeddings of images."""
        B = len(images)
        if not evaluation and self.train_backbone:
            self.forward_modules["feature_aggregator"].train()
            features = self.forward_modules["feature_aggregator"](images, eval=evaluation)
        else:
            _ = self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        # Reshape features if necessary (e.g., from (B, L, C) to (B, C, W, H))
        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)

        # Patchify features
        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        # Interpolate patches to a common size
        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]
        
        # Preprocess and aggregate features
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)
        
        aux_loss = None
        # Apply enhanced or original projection layer
        if self.use_enhanced_projection and self.advanced_projection is not None:
            if not evaluation:
                features, aux_loss = self.advanced_projection(features, add_noise=True)
            else:
                with torch.no_grad():
                    features, _ = self.advanced_projection(features, add_noise=False)
        elif self.pre_projection is not None:
            if not evaluation:
                features = self.pre_projection(features)
            else:
                with torch.no_grad():
                    features = self.pre_projection(features)
        
        return features, patch_shapes, aux_loss

    def _train_discriminator(self, input_data):
        """Trains the discriminator and projection layers."""
        # Set feature aggregation modules to evaluation mode
        _ = self.forward_modules.eval()
        
        # Set projection and discriminator to train mode
        if self.use_enhanced_projection and self.advanced_projection is not None:
            self.advanced_projection.train()
        elif self.pre_projection is not None:
            self.pre_projection.train()
        self.discriminator.train()
        
        LOGGER.info(f"Training enhanced discriminator...")
        with tqdm.tqdm(total=self.gan_epochs) as pbar:
            for i_epoch in range(self.gan_epochs):
                all_loss = []
                all_p_true = []
                all_p_fake = []
                
                for data_item in input_data:
                    self.dsc_opt.zero_grad()
                    if self.proj_opt is not None:
                        self.proj_opt.zero_grad()

                    img = data_item["image"]
                    img = img.to(torch.float).to(self.device)
                    
                    # Get true features from normal images
                    true_feats, _, aux_loss = self._embed(img, evaluation=False)
                    
                    # Generate noisy (fake) features
                    noise_idxs = torch.randint(0, self.mix_noise, torch.Size([true_feats.shape[0]]))
                    noise_one_hot = torch.nn.functional.one_hot(noise_idxs, num_classes=self.mix_noise).to(self.device)
                    noise = torch.stack([
                        torch.normal(0, self.noise_std * 1.1**(k), true_feats.shape)
                        for k in range(self.mix_noise)], dim=1).to(self.device)
                    noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)
                    fake_feats = true_feats + noise

                    # Get discriminator scores for true and fake features
                    scores, current_margin = self.discriminator(torch.cat([true_feats, fake_feats]))
                    
                    true_scores = scores[:len(true_feats)]
                    fake_scores = scores[len(fake_feats):]
                    
                    th = current_margin # Current threshold/margin for hinge loss
                    
                    # Calculate hinge loss for discriminator
                    true_loss = torch.clip(-true_scores + th, min=0).mean()
                    fake_loss = torch.clip(fake_scores + th, min=0).mean()

                    # Monitor true/fake prediction rates
                    p_true = (true_scores.detach() >= th).float().mean()
                    p_fake = (fake_scores.detach() < -th).float().mean()

                    # Log metrics to TensorBoard
                    self.logger.logger.add_scalar(f"p_true", p_true, self.logger.g_iter)
                    self.logger.logger.add_scalar(f"p_fake", p_fake, self.logger.g_iter)
                    self.logger.logger.add_scalar(f"current_margin", current_margin.item(), self.logger.g_iter)

                    loss = true_loss + fake_loss
                    if aux_loss is not None and aux_loss.item() > 0:
                        loss += aux_loss
                        self.logger.logger.add_scalar("aux_loss", aux_loss.item(), self.logger.g_iter)

                    self.logger.logger.add_scalar("loss", loss, self.logger.g_iter)
                    self.logger.step()

                    # Backpropagate and update weights
                    loss.backward()
                    if self.proj_opt is not None:
                        self.proj_opt.step()
                    if self.train_backbone:
                        self.backbone_opt.step()
                    self.dsc_opt.step()

                    loss = loss.detach().cpu() 
                    all_loss.append(loss.item())
                    all_p_true.append(p_true.cpu().item())
                    all_p_fake.append(p_fake.cpu().item())
                
                if self.cos_lr:
                    self.dsc_schl.step()
                
                # Log epoch-wise averages
                all_loss = sum(all_loss) / len(input_data)
                all_p_true = sum(all_p_true) / len(input_data)
                all_p_fake = sum(all_p_fake) / len(input_data)
                cur_lr = self.dsc_opt.state_dict()['param_groups'][0]['lr']
                pbar_str = f"epoch:{i_epoch} loss:{round(all_loss, 5)} "
                pbar_str += f"lr:{round(cur_lr, 6)}"
                pbar_str += f" p_true:{round(all_p_true, 3)} p_fake:{round(all_p_fake, 3)}"
                pbar.set_description_str(pbar_str)
                pbar.update(1)

    def train(self, training_data, test_data):
        """Trains the MAADNet model."""
        state_dict = {}
        ckpt_path = os.path.join(self.ckpt_dir, "ckpt.pth")
        
        # Load existing checkpoint if available to resume training or evaluate
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=self.device)
            if 'discriminator' in state_dict:
                self.discriminator.load_state_dict(state_dict['discriminator'])
                if self.use_enhanced_projection and "advanced_projection" in state_dict:
                    self.advanced_projection.load_state_dict(state_dict["advanced_projection"])
                elif self.pre_proj > 0 and "pre_projection" in state_dict:
                    self.pre_projection.load_state_dict(state_dict["pre_projection"])
            else:
                self.load_state_dict(state_dict, strict=False)

            # Evaluate with loaded model
            scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
            auroc, full_pixel_auroc, anomaly_pixel_auroc = self._evaluate(test_data, scores, segmentations, features, labels_gt, masks_gt)
            
            return auroc, full_pixel_auroc, anomaly_pixel_auroc
        
        def update_state_dict(d):
            """Helper to update model state dictionary for saving."""
            d["discriminator"] = OrderedDict({
                k:v.detach().cpu() 
                for k, v in self.discriminator.state_dict().items()})
            if self.use_enhanced_projection and self.advanced_projection is not None:
                d["advanced_projection"] = OrderedDict({
                    k:v.detach().cpu() 
                    for k, v in self.advanced_projection.state_dict().items()})
            elif self.pre_projection is not None:
                d["pre_projection"] = OrderedDict({
                    k:v.detach().cpu() 
                    for k, v in self.pre_projection.state_dict().items()})

        best_record = None
        for i_mepoch in range(self.meta_epochs):

            self._train_discriminator(training_data)

            # Evaluate after each meta-epoch
            scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
            auroc, full_pixel_auroc, pro = self._evaluate(test_data, scores, segmentations, features, labels_gt, masks_gt)
            self.logger.logger.add_scalar("i-auroc", auroc, i_mepoch)
            self.logger.logger.add_scalar("p-auroc", full_pixel_auroc, i_mepoch)
            self.logger.logger.add_scalar("pro", pro, i_mepoch)

            # Update best record and save checkpoint if performance improves
            if best_record is None:
                best_record = [auroc, full_pixel_auroc, pro]
                update_state_dict(state_dict)
            else:
                if auroc > best_record[0]:
                    best_record = [auroc, full_pixel_auroc, pro]
                    update_state_dict(state_dict)
                elif auroc == best_record[0] and full_pixel_auroc > best_record[1]:
                    best_record[1] = full_pixel_auroc
                    best_record[2] = pro 
                    update_state_dict(state_dict)

            print(f"----- {i_mepoch} I-AUROC:{round(auroc, 4)}(MAX:{round(best_record[0], 4)})"
                  f"  P-AUROC{round(full_pixel_auroc, 4)}(MAX:{round(best_record[1], 4)}) -----"
                  f"  PRO-AUROC{round(pro, 4)}(MAX:{round(best_record[2], 4)}) -----")
        
        torch.save(state_dict, ckpt_path)
        return best_record

    def _predict(self, images):
        """Infers anomaly scores and masks for a batch of images."""
        images = images.to(torch.float).to(self.device)
        # Set feature aggregation modules to evaluation mode
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        # Set projection and discriminator to evaluation mode
        if self.use_enhanced_projection and self.advanced_projection is not None:
            self.advanced_projection.eval()
        elif self.pre_projection is not None:
            self.pre_projection.eval()
        self.discriminator.eval()
        
        with torch.no_grad():
            features, patch_shapes, _ = self._embed(images, provide_patch_shapes=True, evaluation=True)

            scores, _ = self.discriminator(features)
            
            # MAADNet uses negative discriminator scores as anomaly scores
            patch_scores = -scores
            image_scores = -scores

            patch_scores = patch_scores.cpu().numpy()
            image_scores = image_scores.cpu().numpy()

            # Unpatch and aggregate image scores
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            # Unpatch and reshape patch scores for segmentation
            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
            features = features.reshape(batchsize, scales[0], scales[1], -1)
            masks, features = self.anomaly_segmentor.convert_to_segmentation(patch_scores, features)

        return list(image_scores), list(masks), list(features)

    def predict(self, data, prefix=""):
        """Provides anomaly scores/maps for given data (DataLoader or single batch)."""
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data, prefix)
        return self._predict(data)

    def _predict_dataloader(self, dataloader, prefix):
        """Provides anomaly scores/maps for a full dataloader."""
        _ = self.forward_modules.eval()

        img_paths = []
        scores = []
        masks = []
        features = []
        labels_gt = []
        masks_gt = []

        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    labels_gt.extend(data["is_anomaly"].numpy().tolist())
                    if data.get("mask", None) is not None:
                        masks_gt.extend(data["mask"].numpy().tolist())
                    image = data["image"]
                    img_paths.extend(data['image_path'])
                _scores, _masks, _feats = self._predict(image)
                for score, mask, feat, is_anomaly in zip(_scores, _masks, _feats, data["is_anomaly"].numpy().tolist()):
                    scores.append(score)
                    masks.append(mask)

        return scores, masks, features, labels_gt, masks_gt

    def _evaluate(self, test_data, scores, segmentations, features, labels_gt, masks_gt):
        """Evaluates model performance using various metrics (AUROC, PRO)."""
        scores = np.squeeze(np.array(scores))
        # Normalize image scores to [0, 1]
        img_min_scores = scores.min(axis=-1)
        img_max_scores = scores.max(axis=-1)
        scores = (scores - img_min_scores) / (img_max_scores - img_min_scores)

        # Compute image-wise AUROC
        auroc = metrics.compute_imagewise_retrieval_metrics(
            scores, labels_gt 
        )["auroc"]

        if len(masks_gt) > 0:
            segmentations = np.array(segmentations)
            # Normalize pixel-wise segmentations to [0, 1]
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            norm_segmentations = np.zeros_like(segmentations)
            for min_score, max_score in zip(min_scores, max_scores):
                norm_segmentations += (segmentations - min_score) / max(max_score - min_score, 1e-2)
            norm_segmentations = norm_segmentations / len(scores)

            # Compute pixel-wise AUROC
            pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
                norm_segmentations, masks_gt)
            full_pixel_auroc = pixel_scores["auroc"]

            # Compute PRO score
            pro = metrics.compute_pro(np.squeeze(np.array(masks_gt)), 
                                            norm_segmentations)
        else:
            full_pixel_auroc = -1 # Indicate not applicable if no masks
            pro = -1 # Indicate not applicable if no masks

        return auroc, full_pixel_auroc, pro

    def test(self, training_data, test_data, save_segmentation_images):
        """Tests the MAADNet model, optionally saving segmentation images."""
        ckpt_path = os.path.join(self.ckpt_dir, "ckpt.pth")
        # Load model checkpoint
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=self.device)
            if self.use_enhanced_projection and "advanced_projection" in state_dict:
                self.advanced_projection.load_state_dict(state_dict["advanced_projection"])
            elif self.pre_proj > 0 and "pre_projection" in state_dict:
                self.pre_projection.load_state_dict(state_dict["pre_projection"])
            if "discriminator" in state_dict:
                self.discriminator.load_state_dict(state_dict["discriminator"])

        aggregator = {"scores": [], "segmentations": [], "features": []}
        scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
        aggregator["scores"].append(scores)
        aggregator["segmentations"].append(segmentations)
        aggregator["features"].append(features)

        # Aggregate and normalize scores/segmentations across multiple runs (if applicable)
        scores = np.array(aggregator["scores"])
        min_scores = scores.min(axis=-1).reshape(-1, 1)
        max_scores = scores.max(axis=-1).reshape(-1, 1)
        scores = (scores - min_scores) / (max_scores - min_scores)
        scores = np.mean(scores, axis=0)

        segmentations = np.array(aggregator["segmentations"])
        min_scores = (
            segmentations.reshape(len(segmentations), -1)
            .min(axis=-1)
            .reshape(-1, 1, 1, 1)
        )
        max_scores = (
            segmentations.reshape(len(segmentations), -1)
            .max(axis=-1)
            .reshape(-1, 1, 1, 1)
        )
        segmentations = (segmentations - min_scores) / (max_scores - min_scores)
        segmentations = np.mean(segmentations, axis=0)

        # Determine anomaly labels from dataset
        anomaly_labels = [
            x[1] != "good" for x in test_data.dataset.data_to_iterate
        ]

        if save_segmentation_images:
            self.save_segmentation_images(test_data, segmentations, scores)
            
        # Compute final evaluation metrics
        auroc = metrics.compute_imagewise_retrieval_metrics(
            scores, anomaly_labels
        )["auroc"]

        pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
            segmentations, masks_gt
        )
        full_pixel_auroc = pixel_scores["auroc"]
        
        pro = metrics.compute_pro(np.squeeze(np.array(masks_gt)), 
                                segmentations)

        return auroc, full_pixel_auroc, pro

    @staticmethod
    def _params_file(filepath, prepend=""):
        """Returns path to parameters pickle file."""
        return os.path.join(filepath, prepend + "params.pkl")

    def save_to_path(self, save_path: str, prepend: str = ""):
        """Saves model parameters (Note: Model state is saved in train method)."""
        LOGGER.info("Saving data. (Note: Model state is saved in train method.)")
        pass

    def save_segmentation_images(self, data, segmentations, scores):
        """Saves anomaly segmentation images to disk."""
        image_paths = [
            x[2] for x in data.dataset.data_to_iterate
        ]
        mask_paths = [
            x[3] for x in data.dataset.data_to_iterate
        ]

        def image_transform(image):
            """Transforms image for visualization (denormalize and convert to uint8)."""
            in_std = np.array(
                data.dataset.transform_std
            ).reshape(-1, 1, 1)
            in_mean = np.array(
                data.dataset.transform_mean
            ).reshape(-1, 1, 1)
            image = data.dataset.transform_img(image)
            return np.clip(
                (image.numpy() * in_std + in_mean) * 255, 0, 255
            ).astype(np.uint8)

        def mask_transform(mask):
            """Transforms mask for visualization (convert to numpy)."""
            return data.dataset.transform_mask(mask).numpy()

        plot_segmentation_images(
            '/root/MAADNet/seg_img_result',
            image_paths,
            segmentations,
            scores,
            mask_paths,
            image_transform=image_transform,
            mask_transform=mask_transform,
        )
        
        # plot_overlay_segmentation(
        #     savefolder='/root/MAADNet/seg_img_overlay_result',
        #     image_paths=image_paths,
        #     segmentations=segmentations,
        #     anomaly_scores=scores,
        #     image_transform=image_transform, 
        #     alpha=0.4, # 调整透明度，0.4-0.6 通常效果较好
        #     cmap='magma', # 尝试不同的颜色映射，例如 'magma', 'hot', 'viridis', 'jet'
        #     interpolation='bilinear' # 可以尝试 'nearest' 如果您想要更锐利的像素效果
        # )

class Projection(torch.nn.Module):
    """Original projection layer for compatibility."""
    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        super().__init__()
        
        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes 
            self.layers.add_module(f"{i}fc", 
                                   torch.nn.Linear(_in, _out))
            if i < n_layers - 1:
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu",
                                           torch.nn.LeakyReLU(.2))
        self.apply(init_weight)
    
    def forward(self, x):
        return self.layers(x)

class PatchMaker:
    """Generates and processes image patches."""
    def __init__(self, patchsize, top_k=0, stride=None):
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
        """Converts feature tensors into patches."""
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        """Reshapes patch scores back to batch format."""
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        """Aggregates patch scores to image-level scores."""
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 2:
            x = torch.max(x, dim=-1).values
        if x.ndim == 2:
            if self.top_k > 1:
                x = torch.topk(x, self.top_k, dim=1).values.mean(1)
            else:
                x = torch.max(x, dim=1).values
        if was_numpy:
            return x.numpy()
        return x
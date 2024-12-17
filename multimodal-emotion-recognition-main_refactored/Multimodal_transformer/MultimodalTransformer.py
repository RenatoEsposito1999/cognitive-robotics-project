import torch
import torch.nn as nn

from Multimodal_transformer.Preprocessing_CNN.Audio_preprocessing import AudioCNNPool
from Multimodal_transformer.Preprocessing_CNN.EEG_preprocessing import EEGCNNPreprocessor
from Multimodal_transformer.Preprocessing_CNN.Video_preprocessing import EfficientFaceTemporal
from Multimodal_transformer.CNN_nostro.EEG_preprocessing_cnn import EEG_cnn
# Da eliminare EEGTransformerEncoder
from Multimodal_transformer.Transformers.Transformer_funcs import EEGTransformerEncoder, AttentionBlock

from Multimodal_transformer.autoencoder import EEGAutoencoder
from Multimodal_transformer.EEG_spatial_extraction import SpatialFeatureExtractor
from Multimodal_transformer.EEG_temporal_extraction import TemporalFeatureExtractor


class MultimodalTransformer(nn.Module):
    def __init__(self, num_classes=4, seq_length=15, num_channels_eeg=14, pretr_ef='None', num_heads=1):
        super(MultimodalTransformer, self).__init__()

        self.embeds_dim = 128
        self.num_channels_eeg = num_channels_eeg

        self.audio_preprocessing = AudioCNNPool(num_classes=num_classes)
        self.video_preprocessing = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], num_classes, seq_length)
        #self.EEG_preprocessing = EEGCNNPreprocessor(d_model=self.embeds_dim, num_channels=self.num_channels_eeg, cnn_out_channels=self.embeds_dim)
        self.EEG_preprocessing_nostro = EEG_cnn(num_channels=self.num_channels_eeg, output_features=self.embeds_dim, num_band=5)
        self.av = AttentionBlock(
            in_dim_k=self.embeds_dim, in_dim_q=self.embeds_dim, out_dim=self.embeds_dim, num_heads=num_heads
        )

        self.va = AttentionBlock(
            in_dim_k=self.embeds_dim, in_dim_q=self.embeds_dim, out_dim=self.embeds_dim, num_heads=num_heads
        )

        #self.EEG_Transformer = EEGTransformerEncoder(d_model=self.embeds_dim, num_heads=num_heads)

        self.EEG_CrossAttention = AttentionBlock(
            in_dim_k=self.embeds_dim, in_dim_q=self.embeds_dim, out_dim=self.embeds_dim, num_heads=num_heads
        )

        self.Layer_norm = nn.LayerNorm(self.embeds_dim * 3)
        self.fc = nn.Linear(self.embeds_dim * 3, num_classes)

        self.eeg_aux_classifier = nn.Linear(self.embeds_dim, num_classes)

        # Learnable parameters for modality weights (as a convex combination)
        self.modality_weights = nn.Parameter(torch.FloatTensor([1.0, 1.0, 3.0]))
        
        weights = torch.load("/home/v.mele/cognitive_robotics/Multimoda_testing_1/Multimodal_transformer/EEG_preprocessing_AE/best_autoencoder.pth")
        self.autoencoder = EEGAutoencoder(latent_channels=128)
        self.autoencoder.load_state_dict(weights["state_dict"])
        
        self.spatial_extractor = SpatialFeatureExtractor (self.num_channels_eeg, 5, output_dim=self.embeds_dim)
        self.temporal_extractor = TemporalFeatureExtractor(input_dim=self.embeds_dim)

    def forward(self, x_audio, x_visual, x_eeg, device):
        """
        Forward pass for the multimodal transformer model.
        Args:
            x_audio: Audio input tensor
            x_visual: Video input tensor
            x_eeg: EEG input tensor
            device: Device to use (CPU/GPU)

        Returns:
            main_output: Final logits combining all modalities
            eeg_aux_output: Auxiliary output for EEG modality
        """

        
        # Process audio
        x_audio = self.audio_preprocessing.forward_stage1(x_audio)
        proj_x_a = self.audio_preprocessing.forward_stage2(x_audio)

        # Process video
        x_visual = self.video_preprocessing.forward_features(x_visual)
        x_visual = self.video_preprocessing.forward_stage1(x_visual)
        proj_x_v = self.video_preprocessing.forward_stage2(x_visual)
       
        # Process EEG
        #proj_x_eeg = self.EEG_preprocessing_nostro.forward(x_eeg)
        '''with torch.no_grad():
            proj_x_eeg, _ = self.autoencoder(x_eeg)'''
        
        x_eeg = x_eeg.permute(0,2,1,3)
        batch_size, seq_length, num_channels, num_bands = x_eeg.size()
        x = x_eeg.view(batch_size * seq_length, 1, num_channels, num_bands)  # Merge batch and seq_length
        x = self.spatial_extractor(x)  # Output: [batch_size * seq_length, output_dim]
        x = x.view(seq_length, batch_size, -1)  # Reshape to [seq_length, batch_size, output_dim]
        proj_x_eeg = self.temporal_extractor(x)  # Output: [seq_length, batch_size, output_dim]
        
        
        # Attaccare qua l'autoencoder (encoding)
        proj_x_eeg = proj_x_eeg.permute(1,0,2)
        # Permute audio and video projections
        proj_x_a = proj_x_a.permute(0, 2, 1)
        proj_x_v = proj_x_v.permute(0, 2, 1)

        # Cross-attention between audio and video
        audio_video_combined = self.av(proj_x_v, proj_x_a)
        video_audio_combined = self.va(proj_x_a, proj_x_v)

        # Pool audio and video features
        audio_pooled = audio_video_combined.mean(dim=1)
        video_pooled = video_audio_combined.mean(dim=1)

        # Pool EEG features
        combined_audio_video = torch.cat((audio_video_combined, video_audio_combined), dim=1)
        eeg_attended = self.EEG_CrossAttention(combined_audio_video, proj_x_eeg)
        eeg_pooled = eeg_attended.mean(dim=1)
        

        # Apply softmax to modality weights to ensure convex combination
        normalized_weights = torch.softmax(self.modality_weights, dim=0)

        # Weighted combination of modalities
        # Define if necessary to tune the model or keep constant the importance of the inputs
        concat_audio_video_eeg = torch.cat((
            normalized_weights[0] * audio_pooled,
            normalized_weights[1] * video_pooled,
            normalized_weights[2] * eeg_pooled
        ), dim=-1)

        # Layer normalization and final classification
        concat_audio_video_eeg = self.Layer_norm(concat_audio_video_eeg.to(device))
        main_output = self.fc(concat_audio_video_eeg)

        return main_output
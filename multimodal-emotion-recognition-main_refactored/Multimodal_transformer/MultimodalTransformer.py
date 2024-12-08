import torch
import torch.nn as nn

from Multimodal_transformer.Preprocessing_CNN.Audio_preprocessing import AudioCNNPool
from Multimodal_transformer.Preprocessing_CNN.EEG_preprocessing import EEGCNNPreprocessor
from Multimodal_transformer.Preprocessing_CNN.Video_preprocessing import EfficientFaceTemporal

from Multimodal_transformer.Transformers.Transformer_funcs import EEGTransformerEncoder,AttentionBlock

class MultimodalTransformer(nn.Module):
    def __init__(self,num_classes=4,seq_length=15,num_channels_eeg=14,pretr_ef='None',num_heads=1):
        super(MultimodalTransformer,self).__init__()

        self.embeds_dim = 128
        self.num_channels_eeg = num_channels_eeg

        self.audio_preprocessing = AudioCNNPool(num_classes=num_classes)
        self.video_preprocessing = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], num_classes, seq_length)
        self.EEG_preprocessing = EEGCNNPreprocessor(d_model=self.embeds_dim, num_channels=self.num_channels_eeg, cnn_out_channels=self.embeds_dim)

        self.av = AttentionBlock(
            in_dim_k=self.embeds_dim, in_dim_q=self.embeds_dim, out_dim=self.embeds_dim, num_heads=num_heads
        )

        self.VideoToAudio_CrossAttention = AttentionBlock(
            in_dim_k=self.embeds_dim, in_dim_q=self.embeds_dim, out_dim=self.embeds_dim, num_heads=num_heads
        )

        self.EEG_Transformer = EEGTransformerEncoder(d_model=self.embeds_dim,num_heads=num_heads)

        self.EEG_CrossAttention = AttentionBlock(
            in_dim_k=self.embeds_dim, in_dim_q=self.embeds_dim, out_dim=self.embeds_dim, num_heads=num_heads
        )
        
        self.Layer_norm = nn.LayerNorm(self.embeds_dim*3)
        self.fc = nn.Linear(self.embeds_dim*3,num_classes)

        self.eeg_aux_classifier = nn.Linear(self.embeds_dim, num_classes)

        self.modality_weights = nn.Parameter(torch.ones(3))
        
    def forward(self,x_audio,x_visual,x_eeg, device):

        '''
        This procedure utilizes cross-attention between audio and video to produce the embedding of the transformer. Those
        embeds are then again attentioned through an additional cross-attention step to produce a final output used to 
        syncronize each modality. The main elements which are included in this procedure are:
        - Introduction of learnable weights respect to the various embeds produced.
        - Layer normalization across the concatenation to ensure proper classification task.
        
        Output:
        - main_output: logits produced by the classification obtained on the concatenation of sources.
        - eeg_aux_output: Auxiliary output(to compute an additional loss term) derived from the EEG modality in order to identify potential issues arised by this
        element in the attention mechanism.

        NOTE: In this final implementation of the cross-attetion across all the possible modalities the EEG_transformer is ignored
        to be substituted with the same transformer attention mechanism utilized for the other two sources.
        '''

        x_audio = self.audio_preprocessing.forward_stage1(x_audio)
        proj_x_a = self.audio_preprocessing.forward_stage2(x_audio)

        x_visual = self.video_preprocessing.forward_features(x_visual) 
        x_visual = self.video_preprocessing.forward_stage1(x_visual)
        proj_x_v = self.video_preprocessing.forward_stage2(x_visual)

        proj_x_eeg= self.EEG_preprocessing.forward(x_eeg)

        proj_x_a = proj_x_a.permute(0, 2, 1)
        proj_x_v = proj_x_v.permute(0, 2, 1)
        
        h_av = self.av(proj_x_v, proj_x_a)
        h_va = self.va(proj_x_a, proj_x_v)

        audio_pooled = h_av.mean([1]) #mean accross temporal dimension
        video_pooled = h_va.mean([1])
          
        eeg_features = self.EEG_Transformer.forward(proj_x_eeg, device)

        eeg_aux_output = self.eeg_aux_classifier(eeg_features)

        # New block to introduce cross attention between audio-video and eeg
        combined_audio_video = torch.cat((h_av, h_va), dim=1)
        # Treat EEG as query, and combined video-audio as key-value
        eeg_attended = self.EEG_CrossAttention(combined_audio_video, proj_x_eeg)
        eeg_pooled = eeg_attended.mean(dim=1)

        # Learnable parameters for the importance of each modality (debugging utility + reguralization)
        concat_audio_video_eeg = torch.cat((
            self.modality_weights[0] * audio_pooled,
            self.modality_weights[1] * video_pooled,
            self.modality_weights[2] * eeg_pooled
        ), dim=-1)

        concat_audio_video_eeg = self.Layer_norm(concat_audio_video_eeg.to(device))
        
        main_output = self.fc(concat_audio_video_eeg)
        
        return main_output,eeg_aux_output
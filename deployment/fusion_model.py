import torch
import torch.nn as nn

class LateFusionModel(nn.Module):
    """
    Late fusion for any combination of Audio / Video / RF.
    - Each backbone output -> normalised probability [B, 1]
    - Concatenated -> [B, num_active]
    - Linear(num_active,1) + Sigmoid -> [B]
    - MODIFIED: Also returns individual modality probabilities
    """

    def __init__(self, model_audio=None, model_visual=None, model_rf=None,
                 num_classes=2):
        super(LateFusionModel, self).__init__()

        self.audio = model_audio
        self.visual = model_visual
        self.rf = model_rf

        num_active = sum([model_audio is not None,
                          model_visual is not None,
                          model_rf is not None])
        if num_active == 0:
            raise ValueError('At least one backbone must be provided.')

        self.final_pred = nn.Sequential(
            nn.Linear(num_active, num_classes - 1),
            nn.Sigmoid()
        )

    @staticmethod
    def _get_prob(model_out):
        """Normalise backbone output to [B, 1] probability."""
        if isinstance(model_out, (list, tuple)):
            model_out = model_out[-1]
        
        # Handle different output shapes
        if model_out.dim() == 0:  # scalar
            model_out = model_out.unsqueeze(0).unsqueeze(0)
        elif model_out.dim() == 1:  # [B]
            model_out = model_out.unsqueeze(1)
        elif model_out.dim() == 2 and model_out.size(1) != 1:
            model_out = model_out.view(model_out.size(0), -1)
        else:
            model_out = model_out.view(model_out.size(0), 1)
        
        if model_out.min() < 0.0 or model_out.max() > 1.0:
            model_out = torch.sigmoid(model_out)
        return model_out

    def forward(self, audio=None, video=None, rf=None, return_individual=False):
        """
        Forward pass
        
        Args:
            audio: Audio tensor [B, 1, 64, time_frames] or None
            video: Video tensor [B, 7, 3, 112, 112] or None
            rf: RF tensor [B, 1, 3, 112, 112] or None
            return_individual: If True, returns dict with individual scores + fusion
        
        Returns:
            If return_individual=False: Tensor of fusion predictions [B]
            If return_individual=True: Dict with:
                - 'fusion': Tensor of fusion predictions [B]
                - 'individual': Dict of modality name -> probability tensor [B]
                - 'modalities_used': List of modalities that were provided
        """
        probs = []
        modalities_used = []

        if self.audio is not None and audio is not None:
            probs.append(self._get_prob(self.audio(audio)))
            modalities_used.append('audio')

        if self.visual is not None and video is not None:
            probs.append(self._get_prob(self.visual(video)))
            modalities_used.append('video')

        if self.rf is not None and rf is not None:
            if rf.dim() == 5:
                rf = rf[:, 0]
            probs.append(self._get_prob(self.rf(rf)))
            modalities_used.append('rf')

        # If no modalities provided
        if len(probs) == 0:
            if return_individual:
                return {'fusion': None, 'individual': {}, 'modalities_used': []}
            return None

        # Combine for fusion prediction
        combined = torch.cat(probs, dim=-1)
        fusion_pred = self.final_pred(combined).squeeze()

        if return_individual:
            return {
                'fusion': fusion_pred,
                'individual': {
                    mod: prob.squeeze() for mod, prob in zip(modalities_used, probs)
                },
                'modalities_used': modalities_used
            }
        return fusion_pred
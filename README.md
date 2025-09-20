# MaskFreeAVSE4AVSR
Codes for "Purification Before Fusion: Toward Mask-Free Speech Enhancement for Robust Audio-Visual Speech Recognition"
+ Multimodal bottleneck Conformer (MBT-denoiser)
+ Audio-visual speech enhancement using clean mel-spectrogram reconstruction (L1) and [perceptual loss](https://github.com/adrienchaton/PerceptualAudio_pytorch) (MSE)
+ Multimodal Conformer encoder (inter-modal and intra-modal fusion)

Introduction: 
We propose a noise-robust AVSR framework that eliminates the need for explicit mask-based denoising. By leveraging a Multimodal Bottleneck Conformer and reconstruction-based objectives, our approach achieves superior performance under noisy conditions. This work presents a mask-free framework for noise-robust AVSR that addresses the semantic information loss problem inherent in explicit masking approaches. The key contributions include the Multimodal Bottleneck Conformer architecture and the reconstruction-based training strategy that purifies noisy audio representations through visual guidance.

# Bridging the Data Gap: Spatially Conditioned Diffusion Model for Anomaly Generation in Photovoltaic Electroluminescence Images

<img width="1268" height="712" alt="image" src="https://github.com/user-attachments/assets/7fa0899c-dc6b-4ac9-aa72-c222508f758d" />


## Abstract
Reliable anomaly detection (AD) in photovoltaic (PV) modules is critical for maintaining solar energy efficiency. However, the development 
of robust computer vision models for PV inspection is limited by the scarcity of large-scale, diverse, and balanced datasets. In this study,
we introduce a spatially conditioned denoising diffusion probabilistic model, PV-DDPM, designed to generate anomalous electroluminescence (EL) images across four PV cell types: 
multi-crystalline silicon (multi-c-Si), mono-crystalline silicon (mono-c-Si), half-cut multi-c-Si, and interdigitated back contact (IBC) with dogbone interconnect. PV-DDPM enables
the controlled synthesis of both multi-defect and single-defect scenarios by conditioning on binary masks that represent structural features and defect positions. To the bestof our
knowledge, this is the first framework that jointly models multiple PV cell types while allowing simultaneous generation of diverse anomaly types. We also introduce E-SCDD dataset,
an enhanced version of the SCDD dataset, comprising 1,000 pixel-wise annotated EL images spanning30 semantic classes, and 1,768 unlabeled synthetic samples.Quantitative evaluation 
shows our generated images achievea Frechet Inception Distance (FID) of 4.10 and Kernel Inception Distance (KID) of 0.0023 ± 0.0007 across all categories. Furthermore, compared to 
training on the original SCDD dataset, training the vision–language anomaly detection model AA-CLIP on E-SCDD improves pixel-level AUC and average precision by 1.70 and 8.34 points,
respectively.

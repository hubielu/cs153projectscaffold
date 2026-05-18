A PyTorch encoder-decoder that takes raw current-voltage (IV) measurements from 2D-semiconductor field-effect transistors and produces **(a) the underlying device parameters** that explain the measurements and **(b) a reconstruction of the IV curves** themselves. 

The same model is trained jointly across multiple materials and device geometries so the encoder learns a **material-agnostic latent representation** of *"what this device looks like electrically."* This matters because parameter extraction for next-generation 2D FETs is currently done one material and one geometry at a time, with bespoke forward/inverse networks per setup. A single conditioning-based encoder-decoder would:
* **Cut the per-material training cost**
* **Let researchers transfer to new materials** with little data
* **Provide a reconstruction-error signal** that doubles as an unsupervised anomaly detector for flagging faulty fabricated devices before destructive testing.

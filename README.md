# PLINK
**P**robabilistic **Li**DAR **N**eRF **K**odebase

## Novel Viewpoint Generation
PLINK can be used to generate a high quality synthetic LiDAR scans from previously unvisited locations. PLINK can learn a robust scene representation from messy raw data by reformulating the NeRF training routine to better fit the properties of LiDAR point cloud data.   

<div style="display: flex; justify-content: space-between;">
  <div style="text-align: center; margin-right: 20px;">
    <div><strong>Raw LiDAR data used to train NeRF</strong></div>
    <img src="./demo/trainingDataCourtyard.gif" width="400" />
  </div>
  <div style="text-align: center;">
    <div><strong>Synthetic LiDAR Scans Generated by NeRF</strong></div>
    <img src="./demo/NCv14.gif" width="400" />
  </div>
</div>


# TODO List

- [X] Set up project repository
- [X] Make Demo Gifs on Newer College Courtyard Scene
- [ ] Make training timelapse GIF
- [ ] Benchmark performance on courtyard
- [ ] Embed 1% vs 95% CDF figure
- [ ] Ablation study in forest(?)
- [ ] Share interacive jupyter notebook demo
- [ ] Sanitize utils functions
- [ ] Update documentation
- [ ] Add citation link
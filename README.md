# 4DRadar-Simulation
***4DRadar-Simulation*** is Python implementation for synthesizing IF signals of FMCW MIMO Radar
(Frequency-Modulated-Continuous-Wave Multiple-Input-Multiple-Output Radar).

## Beam pattern of ULA
<img src="./images/01_coordinate_system_of_ULA.svg" width="300" style="background-color: white;">

This notebook evaluates beam pattern of uniform linear array (ULA).
see more detail for [01_beam_pattern](./Python/01_beam_pattern.ipynb).

## DoA of ULA
<img src="images/02_range_doppler_map.png" height="200">
<img src="images/02_range_azimuth_map_bf.png" height="200">
<img src="images/02_range_azimuth_map_MUSIC.png" height="200">

This notebook demonstrates DoA method of uniform linear array (ULA).
The FMCW-MIMO radar considered here operates in the 77 GHz frequency band.
see more detail for [02_8ula_single_object](Python/02_8ula_single_object.ipynb).

## DoA of URA
<img src="images/03_coordinate_system_of_URA.svg" height="200" style="background-color: white;">
<img src="images/03_angle_spectrum_bf.png" height="200">
<img src="images/03_angle_spectrum_MUSIC.png" height="200">

This notebook demonstrates DoA method of uniform rectangular array (URA).
The FMCW-MIMO radar considered here is same as ULA simulation.
see more detail for [03_16ura_single_object](Python/03_16ura_single_object.ipynb).

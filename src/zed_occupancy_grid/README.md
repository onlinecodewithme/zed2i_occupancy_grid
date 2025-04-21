# ZED 2i Occupancy Grid

This package implements a 2D occupancy grid mapping system using the ZED 2i stereo camera. It processes depth data to build a map of the environment.

## Fixed Issue: Map Not Updating with Camera Movement

The original system had an issue where the occupancy grid map would not update when the camera was moved. The following fixes have been implemented:

1. **Increased Transform Update Frequency**: 
   - TF publishing frequency increased from 10Hz to 30Hz
   - Added continuous small position changes to force grid updates

2. **Ultra-Sensitive Motion Detection**:
   - Reduced position change threshold from 0.001 to 0.0000001
   - Reduced rotation change threshold from 0.001 to 0.0000001

3. **Optimized Grid Updates**:
   - Added synthetic motion detection to ensure continuous updates
   - Reduced temporal filtering to make updates more responsive
   - Added continuous forced grid publishing

## Available Launch Scripts

Multiple launch scripts are provided for different use cases:

### 1. Standard Launch
```bash
./run_occupancy_grid.sh
```
The original run script with improved parameters.

### 2. Optimized Grid (Recommended)
```bash
./run_optimized_grid.sh
```
Optimized version with ultra-sensitive motion detection and continuous map updates.

### 3. Movement Test
```bash
./test_grid_movement.sh
```
Special test script with debugging output to verify the grid updates with camera movement.

### 4. Complete System
```bash
./run_complete_system.sh
```
Full system with all optimizations enabled.

### 5. Camera Only
```bash
./run_zed_camera.sh
```
Launches only the ZED camera node without the occupancy grid for testing camera functionality.

## Troubleshooting

If the occupancy grid still doesn't update properly with camera movement:

1. Try using the `test_grid_movement.sh` script which has enhanced debugging
2. Check the camera's pose is being published (visible in RViz)
3. Verify all TF frames are present (map → odom → zed_camera_link → zed_left_camera_frame)
4. Increase logging verbosity by setting `export RCUTILS_LOGGING_MIN_SEVERITY=DEBUG`

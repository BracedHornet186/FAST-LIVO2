#include "LIVMapper.h"

using std::placeholders::_1;

// Helper to get seconds from ROS 2 time message
double timeToSec(const builtin_interfaces::msg::Time& time) {
    return rclcpp::Time(time).seconds();
}

LIVMapper::LIVMapper(const rclcpp::NodeOptions & options)
    : Node("liv_mapper", options),
      extT(0, 0, 0),
      extR(M3D::Identity())
{
  extrinT.assign(3, 0.0);
  extrinR.assign(9, 0.0);
  cameraextrinT.assign(3, 0.0);
  cameraextrinR.assign(9, 0.0);

  p_pre.reset(new Preprocess());
  p_imu.reset(new ImuProcess());

  // Initialize TF Broadcaster
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  readParameters();
  VoxelMapConfig voxel_config;
  loadVoxelConfig(this, voxel_config);

  visual_sub_map.reset(new PointCloudXYZI());
  feats_undistort.reset(new PointCloudXYZI());
  feats_down_body.reset(new PointCloudXYZI());
  feats_down_world.reset(new PointCloudXYZI());
  pcl_w_wait_pub.reset(new PointCloudXYZI());
  pcl_l_wait_pub.reset(new PointCloudXYZI());
  pcl_wait_pub.reset(new PointCloudXYZI());
  pcl_wait_save.reset(new PointCloudXYZRGB());
  pcl_wait_save_intensity.reset(new PointCloudXYZI());
  voxelmap_manager.reset(new VoxelMapManager(voxel_config, voxel_map));
  vio_manager.reset(new VIOManager());
  
  // Assuming ROOT_DIR is defined via macro or CMake
  root_dir = ROOT_DIR; 
  initializeFiles();
  initializeComponents();
  
  path.header.stamp = this->now();
  path.header.frame_id = "camera_init";
  
  // Initialize Subs/Pubs after params are read
  initializeSubscribersAndPublishers();
}

LIVMapper::~LIVMapper() {}

void LIVMapper::readParameters()
{
  // Helper lambda for cleaner param declaration
  auto declare_and_get = [this](const std::string & name, auto & var, const auto & default_val)
  {
    if (!this->has_parameter(name)) {
      this->declare_parameter(name, default_val);
    }
    this->get_parameter(name, var);
  };

  declare_and_get("common.img_topic", img_topic, std::string("/left_camera/image"));
  declare_and_get("common.lid_topic", lid_topic, std::string("/livox/lidar"));
  declare_and_get("common.imu_topic", imu_topic, std::string("/livox/imu"));
  declare_and_get("common.img_en", img_en, 1);
  declare_and_get("common.lidar_en", lidar_en, 1);
  declare_and_get("common.ros_driver_bug_fix", ros_driver_fix_en, false);

  declare_and_get("vio.normal_en", normal_en, true);
  declare_and_get("vio.inverse_composition_en", inverse_composition_en, false);
  declare_and_get("vio.max_iterations", max_iterations, 5);
  declare_and_get("vio.img_point_cov", IMG_POINT_COV, 100.0);
  declare_and_get("vio.raycast_en", raycast_en, false);
  declare_and_get("vio.exposure_estimate_en", exposure_estimate_en, true);
  declare_and_get("vio.inv_expo_cov", inv_expo_cov, 0.2);
  declare_and_get("vio.grid_size", grid_size, 5);
  declare_and_get("vio.grid_n_height", grid_n_height, 17);
  declare_and_get("vio.patch_pyrimid_level", patch_pyrimid_level, 3);
  declare_and_get("vio.patch_size", patch_size, 8);
  declare_and_get("vio.outlier_threshold", outlier_threshold, 1000.0);

  declare_and_get("time_offset.exposure_time_init", exposure_time_init, 0.0);
  declare_and_get("time_offset.img_time_offset", img_time_offset, 0.0);
  declare_and_get("time_offset.imu_time_offset", imu_time_offset, 0.0);
  declare_and_get("time_offset.lidar_time_offset", lidar_time_offset, 0.0);
  declare_and_get("uav.imu_rate_odom", imu_prop_enable, false);
  declare_and_get("uav.gravity_align_en", gravity_align_en, false);

  declare_and_get("evo.seq_name", seq_name, std::string("01"));
  declare_and_get("evo.pose_output_en", pose_output_en, false);
  declare_and_get("imu.gyr_cov", gyr_cov, 1.0);
  declare_and_get("imu.acc_cov", acc_cov, 1.0);
  declare_and_get("imu.imu_int_frame", imu_int_frame, 3);
  declare_and_get("imu.imu_en", imu_en, false);
  declare_and_get("imu.gravity_est_en", gravity_est_en, true);
  declare_and_get("imu.ba_bg_est_en", ba_bg_est_en, true);

  declare_and_get("preprocess.blind", p_pre->blind, 0.01);
  declare_and_get("preprocess.filter_size_surf", filter_size_surf_min, 0.5);
  declare_and_get("preprocess.hilti_en", hilti_en, false);
  declare_and_get("preprocess.lidar_type", p_pre->lidar_type, (int)AVIA);
  declare_and_get("preprocess.scan_line", p_pre->N_SCANS, 6);
  declare_and_get("preprocess.point_filter_num", p_pre->point_filter_num, 3);
  declare_and_get("preprocess.feature_extract_enabled", p_pre->feature_enabled, false);

  declare_and_get("pcd_save.interval", pcd_save_interval, -1);
  declare_and_get("pcd_save.pcd_save_en", pcd_save_en, false);
  declare_and_get("pcd_save.colmap_output_en", colmap_output_en, false);
  declare_and_get("pcd_save.filter_size_pcd", filter_size_pcd, 0.5);
  
  declare_and_get("extrin_calib.extrinsic_T", extrinT, std::vector<double>());
  declare_and_get("extrin_calib.extrinsic_R", extrinR, std::vector<double>());
  declare_and_get("extrin_calib.Pcl", cameraextrinT, std::vector<double>());
  declare_and_get("extrin_calib.Rcl", cameraextrinR, std::vector<double>());
  
  declare_and_get("debug.plot_time", plot_time, -10.0);
  declare_and_get("debug.frame_cnt", frame_cnt, 6);

  declare_and_get("publish.blind_rgb_points", blind_rgb_points, 0.01);
  declare_and_get("publish.pub_scan_num", pub_scan_num, 1);
  declare_and_get("publish.pub_effect_point_en", pub_effect_point_en, false);
  declare_and_get("publish.dense_map_en", dense_map_en, false);

  p_pre->blind_sqr = p_pre->blind * p_pre->blind;
}

void LIVMapper::initializeComponents() 
{
  downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
  extT << VEC_FROM_ARRAY(extrinT);
  extR << MAT_FROM_ARRAY(extrinR);

  voxelmap_manager->extT_ << VEC_FROM_ARRAY(extrinT);
  voxelmap_manager->extR_ << MAT_FROM_ARRAY(extrinR);

  if (!vk::camera_loader::loadFromRosNs(this, "laserMapping", vio_manager->cam)) 
      throw std::runtime_error("Camera model not correctly specified.");

  vio_manager->grid_size = grid_size;
  vio_manager->patch_size = patch_size;
  vio_manager->outlier_threshold = outlier_threshold;
  vio_manager->setImuToLidarExtrinsic(extT, extR);
  vio_manager->setLidarToCameraExtrinsic(cameraextrinR, cameraextrinT);
  vio_manager->state = &_state;
  vio_manager->state_propagat = &state_propagat;
  vio_manager->max_iterations = max_iterations;
  vio_manager->img_point_cov = IMG_POINT_COV;
  vio_manager->normal_en = normal_en;
  vio_manager->inverse_composition_en = inverse_composition_en;
  vio_manager->raycast_en = raycast_en;
  vio_manager->grid_n_width = grid_n_width;
  vio_manager->grid_n_height = grid_n_height;
  vio_manager->patch_pyrimid_level = patch_pyrimid_level;
  vio_manager->exposure_estimate_en = exposure_estimate_en;
  vio_manager->colmap_output_en = colmap_output_en;
  vio_manager->initializeVIO();

  p_imu->set_extrinsic(extT, extR);
  p_imu->set_gyr_cov_scale(V3D(gyr_cov, gyr_cov, gyr_cov));
  p_imu->set_acc_cov_scale(V3D(acc_cov, acc_cov, acc_cov));
  p_imu->set_inv_expo_cov(inv_expo_cov);
  p_imu->set_gyr_bias_cov(V3D(0.0001, 0.0001, 0.0001));
  p_imu->set_acc_bias_cov(V3D(0.0001, 0.0001, 0.0001));
  p_imu->set_imu_init_frame_num(imu_int_frame);

  if (!imu_en) p_imu->disable_imu();
  if (!gravity_est_en) p_imu->disable_gravity_est();
  if (!ba_bg_est_en) p_imu->disable_bias_est();
  if (!exposure_estimate_en) p_imu->disable_exposure_est();

  slam_mode_ = (img_en && lidar_en) ? LIVO : imu_en ? ONLY_LIO : ONLY_LO;
}

void LIVMapper::initializeFiles() 
{
  if (pcd_save_en && colmap_output_en)
  {
      const std::string folderPath = std::string(ROOT_DIR) + "/scripts/colmap_output.sh";
      
      std::string chmodCommand = "chmod +x " + folderPath;
      
      int chmodRet = system(chmodCommand.c_str());  
      if (chmodRet != 0) {
          std::cerr << "Failed to set execute permissions for the script." << std::endl;
          return;
      }

      int executionRet = system(folderPath.c_str());
      if (executionRet != 0) {
          std::cerr << "Failed to execute the script." << std::endl;
          return;
      }
  }
  if(colmap_output_en) fout_points.open(std::string(ROOT_DIR) + "Log/Colmap/sparse/0/points3D.txt", std::ios::out);
  if(pcd_save_interval > 0) fout_pcd_pos.open(std::string(ROOT_DIR) + "Log/PCD/scans_pos.json", std::ios::out);
  fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), std::ios::out);
  fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), std::ios::out);
}

void LIVMapper::initializeSubscribersAndPublishers() 
{
  // QoS Setup - using BestEffort for High freq data, Reliable for others
  auto qos_lidar = rclcpp::QoS(rclcpp::KeepLast(200000)).best_effort();
  auto qos_default = rclcpp::QoS(rclcpp::KeepLast(100));

  if (p_pre->lidar_type == AVIA) {
      sub_pcl_livox = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(
          lid_topic, qos_lidar, std::bind(&LIVMapper::livox_pcl_cbk, this, _1));
  } else {
      sub_pcl_std = this->create_subscription<sensor_msgs::msg::PointCloud2>(
          lid_topic, qos_lidar, std::bind(&LIVMapper::standard_pcl_cbk, this, _1));
  }

  sub_imu = this->create_subscription<sensor_msgs::msg::Imu>(
      imu_topic, rclcpp::QoS(rclcpp::KeepLast(200000)), std::bind(&LIVMapper::imu_cbk, this, _1));
  
  sub_img = this->create_subscription<sensor_msgs::msg::Image>(
      img_topic, rclcpp::QoS(rclcpp::KeepLast(200000)), std::bind(&LIVMapper::img_cbk, this, _1));
  
  pubLaserCloudFullRes = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", qos_default);
  pubNormal = this->create_publisher<visualization_msgs::msg::MarkerArray>("visualization_marker", qos_default);
  pubSubVisualMap = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_visual_sub_map_before", qos_default);
  pubLaserCloudEffect = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_effected", qos_default);
  pubLaserCloudMap = this->create_publisher<sensor_msgs::msg::PointCloud2>("/Laser_map", qos_default);
  
  pubOdomAftMapped = this->create_publisher<nav_msgs::msg::Odometry>("/aft_mapped_to_init", 10);
  pubOdomAftMappedCam = this->create_publisher<nav_msgs::msg::Odometry>("/aft_mapped_to_init_cam", 10);
  pubOdomAftMappedLiDAR = this->create_publisher<nav_msgs::msg::Odometry>("/aft_mapped_to_init_lidar", 10);
  pubPath = this->create_publisher<nav_msgs::msg::Path>("/path", 10);
  
  plane_pub = this->create_publisher<visualization_msgs::msg::Marker>("/planner_normal", 1);
  voxel_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>("/voxels", 1);
  pubLaserCloudDyn = this->create_publisher<sensor_msgs::msg::PointCloud2>("/dyn_obj", qos_default);
  pubLaserCloudDynRmed = this->create_publisher<sensor_msgs::msg::PointCloud2>("/dyn_obj_removed", qos_default);
  pubLaserCloudDynDbg = this->create_publisher<sensor_msgs::msg::PointCloud2>("/dyn_obj_dbg_hist", qos_default);
  mavros_pose_publisher = this->create_publisher<geometry_msgs::msg::PoseStamped>("/mavros/vision_pose/pose", 10);
  
  // Image Transport (Requires image_transport plugin for ROS2)
  pubImage = image_transport::create_publisher(this, "/rgb_img");
  pubOriginImage = image_transport::create_publisher(this, "/origin_img");
  
  pubImuPropOdom = this->create_publisher<nav_msgs::msg::Odometry>("/LIVO2/imu_propagate", 10000);
  
  // Timer for IMU propagation
  imu_prop_timer = this->create_wall_timer(
      std::chrono::milliseconds(4), std::bind(&LIVMapper::imu_prop_callback, this));
      
  voxelmap_manager->voxel_map_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/planes", 10000);
  pubLaserCloudFullResBody = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_body", qos_default);
}

void LIVMapper::handleFirstFrame() 
{
  if (!is_first_frame)
  {
    _first_lidar_time = LidarMeasures.last_lio_update_time;
    p_imu->first_lidar_time = _first_lidar_time;
    is_first_frame = true;
    RCLCPP_INFO(this->get_logger(), "FIRST LIDAR FRAME!");
  }
}

void LIVMapper::gravityAlignment() 
{
  if (!p_imu->imu_need_init && !gravity_align_finished) 
  {
    RCLCPP_INFO(this->get_logger(), "Gravity Alignment Starts");
    V3D ez(0, 0, -1), gz(_state.gravity);
    Quaterniond G_q_I0 = Quaterniond::FromTwoVectors(gz, ez);
    M3D G_R_I0 = G_q_I0.toRotationMatrix();

    _state.pos_end = G_R_I0 * _state.pos_end;
    _state.rot_end = G_R_I0 * _state.rot_end;
    _state.vel_end = G_R_I0 * _state.vel_end;
    _state.gravity = G_R_I0 * _state.gravity;
    gravity_align_finished = true;
    RCLCPP_INFO(this->get_logger(), "Gravity Alignment Finished");
  }
}

void LIVMapper::processImu() 
{
  p_imu->Process2(LidarMeasures, _state, feats_undistort);
  if (gravity_align_en) gravityAlignment();

  state_propagat = _state;
  voxelmap_manager->state_ = _state;
  voxelmap_manager->feats_undistort_ = feats_undistort;
}

void LIVMapper::stateEstimationAndMapping() 
{
  switch (LidarMeasures.lio_vio_flg) 
  {
    case VIO:
      handleVIO();
      break;
    case LIO:
    case LO:
      handleLIO();
      break;
  }
}

void LIVMapper::handleVIO() 
{
  euler_cur = RotMtoEuler(_state.rot_end);
  fout_pre << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
            << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
            << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << std::endl;
    
  if (pcl_w_wait_pub->empty() || (pcl_w_wait_pub == nullptr)) 
  {
    RCLCPP_WARN(this->get_logger(), "[ VIO ] No point!!!");
    return;
  }
    
  RCLCPP_INFO(this->get_logger(), "[ VIO ] Raw feature num: %ld", pcl_w_wait_pub->points.size());

  if (fabs((LidarMeasures.last_lio_update_time - _first_lidar_time) - plot_time) < (frame_cnt / 2 * 0.1)) 
  {
    vio_manager->plot_flag = true;
  } 
  else 
  {
    vio_manager->plot_flag = false;
  }

  vio_manager->processFrame(LidarMeasures.measures.back().img, _pv_list, voxelmap_manager->voxel_map_, LidarMeasures.last_lio_update_time - _first_lidar_time);

  if (imu_prop_enable) 
  {
    ekf_finish_once = true;
    latest_ekf_state = _state;
    latest_ekf_time = LidarMeasures.last_lio_update_time;
    state_update_flg = true;
  }

  publish_frame_world(pubLaserCloudFullRes, vio_manager);
  publish_img_rgb(pubImage, vio_manager);

  // full res image
  cv::Mat img_origin = vio_manager->img_origin;
  cv_bridge::CvImage out_msg;
  out_msg.header.stamp = rclcpp::Time(static_cast<int64_t>(LidarMeasures.last_lio_update_time * 1e9));
  out_msg.encoding = sensor_msgs::image_encodings::BGR8;
  out_msg.image = img_origin;
  pubOriginImage.publish(out_msg.toImageMsg());
  publish_odometry(pubOdomAftMapped);
  publish_odometry_cam(pubOdomAftMappedCam);

  euler_cur = RotMtoEuler(_state.rot_end);
  fout_out << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
            << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
            << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << " " << feats_undistort->points.size() << std::endl;
}

void LIVMapper::handleLIO() 
{    
  euler_cur = RotMtoEuler(_state.rot_end);
  fout_pre << setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
           << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
           << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << endl;
           
  if (feats_undistort->empty() || (feats_undistort == nullptr)) 
  {
    RCLCPP_WARN(this->get_logger(), "[ LIO ]: No point!!!");
    return;
  }

  double t0 = omp_get_wtime();

  downSizeFilterSurf.setInputCloud(feats_undistort);
  downSizeFilterSurf.filter(*feats_down_body);
  
  double t_down = omp_get_wtime();

  feats_down_size = feats_down_body->points.size();
  voxelmap_manager->feats_down_body_ = feats_down_body;
  transformLidar(_state.rot_end, _state.pos_end, feats_down_body, feats_down_world);
  voxelmap_manager->feats_down_world_ = feats_down_world;
  voxelmap_manager->feats_down_size_ = feats_down_size;
  
  if (!lidar_map_inited) 
  {
    lidar_map_inited = true;
    voxelmap_manager->BuildVoxelMap();
  }

  double t1 = omp_get_wtime();

  voxelmap_manager->StateEstimation(state_propagat);
  _state = voxelmap_manager->state_;
  _pv_list = voxelmap_manager->pv_list_;

  double t2 = omp_get_wtime();

  if (imu_prop_enable) 
  {
    ekf_finish_once = true;
    latest_ekf_state = _state;
    latest_ekf_time = LidarMeasures.last_lio_update_time;
    state_update_flg = true;
  }

  if (pose_output_en) 
  {
    static bool pos_opend = false;
    std::ofstream evoFile;
    if (!pos_opend) 
    {
      evoFile.open(std::string(ROOT_DIR) + "Log/result/" + seq_name + ".txt", std::ios::out);
      pos_opend = true;
      if (!evoFile.is_open()) RCLCPP_ERROR(this->get_logger(), "open fail");
    } 
    else 
    {
      evoFile.open(std::string(ROOT_DIR) + "Log/result/" + seq_name + ".txt", std::ios::app);
      if (!evoFile.is_open()) RCLCPP_ERROR(this->get_logger(), "open fail");
    }
    Eigen::Quaterniond q(_state.rot_end);
    evoFile << std::fixed;
    evoFile << LidarMeasures.last_lio_update_time << " " << _state.pos_end[0] << " " << _state.pos_end[1] << " " << _state.pos_end[2] << " "
            << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
  }
  
  euler_cur = RotMtoEuler(_state.rot_end);
  // tf::createQuaternionMsgFromRollPitchYaw -> tf2::Quaternion
  tf2::Quaternion q_tf;
  q_tf.setRPY(euler_cur(0), euler_cur(1), euler_cur(2));
  geoQuat = tf2::toMsg(q_tf);

  double t3 = omp_get_wtime();

  PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI());
  transformLidar(_state.rot_end, _state.pos_end, feats_down_body, world_lidar);
  for (size_t i = 0; i < world_lidar->points.size(); i++) 
  {
    voxelmap_manager->pv_list_[i].point_w << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
    M3D point_crossmat = voxelmap_manager->cross_mat_list_[i];
    M3D var = voxelmap_manager->body_cov_list_[i];
    var = (_state.rot_end * extR) * var * (_state.rot_end * extR).transpose() +
          (-point_crossmat) * _state.cov.block<3, 3>(0, 0) * (-point_crossmat).transpose() + _state.cov.block<3, 3>(3, 3);
    voxelmap_manager->pv_list_[i].var = var;
  }
  voxelmap_manager->UpdateVoxelMap(voxelmap_manager->pv_list_);
  RCLCPP_INFO(this->get_logger(), "[ LIO ] Update Voxel Map");
  _pv_list = voxelmap_manager->pv_list_;
  
  double t4 = omp_get_wtime();

  if(voxelmap_manager->config_setting_.map_sliding_en)
  {
    voxelmap_manager->mapSliding();
  }
  
  PointCloudXYZI::Ptr laserCloudFullRes(dense_map_en ? feats_undistort : feats_down_body);
  int size = laserCloudFullRes->points.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

  for (int i = 0; i < size; i++) 
  {
    RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
  }
  *pcl_w_wait_pub = *laserCloudWorld;
  *pcl_l_wait_pub = *laserCloudFullRes;

  // For neural mapping
  sensor_msgs::msg::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*pcl_l_wait_pub, laserCloudmsg);
  laserCloudmsg.header.stamp = rclcpp::Time(static_cast<int64_t>(LidarMeasures.last_lio_update_time * 1e9));
  laserCloudmsg.header.frame_id = "camera_init";
  pubLaserCloudFullResBody->publish(laserCloudmsg);

  publish_odometry_lidar(pubOdomAftMappedLiDAR);

  if (!img_en) publish_frame_world(pubLaserCloudFullRes, vio_manager);
  if (pub_effect_point_en) publish_effect_world(pubLaserCloudEffect, voxelmap_manager->ptpl_list_);
  if (voxelmap_manager->config_setting_.is_pub_plane_map_) voxelmap_manager->pubVoxelMap();
  publish_path(pubPath);
  publish_mavros(mavros_pose_publisher);

  frame_num++;
  aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t4 - t0) / frame_num;

  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m|                         LIO Mapping Time                    |\033[0m\n");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage", "Time (secs)");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "DownSample", t_down - t0);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "ICP", t2 - t1);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "updateVoxelMap", t4 - t3);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Current Total Time", t4 - t0);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Average Total Time", aver_time_consu);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");

  euler_cur = RotMtoEuler(_state.rot_end);
  fout_out << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
            << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
            << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << " " << feats_undistort->points.size() << std::endl;
}

void LIVMapper::savePCD() 
{
  if (pcd_save_en && (pcl_wait_save->points.size() > 0 || pcl_wait_save_intensity->points.size() > 0) && pcd_save_interval < 0) 
  {
    std::string raw_points_dir = std::string(ROOT_DIR) + "Log/PCD/all_raw_points.pcd";
    std::string downsampled_points_dir = std::string(ROOT_DIR) + "Log/PCD/all_downsampled_points.pcd";
    pcl::PCDWriter pcd_writer;

    if (img_en)
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
      voxel_filter.setInputCloud(pcl_wait_save);
      voxel_filter.setLeafSize(filter_size_pcd, filter_size_pcd, filter_size_pcd);
      voxel_filter.filter(*downsampled_cloud);
  
      pcd_writer.writeBinary(raw_points_dir, *pcl_wait_save); 
      std::cout << GREEN << "Raw point cloud data saved to: " << raw_points_dir 
                << " with point count: " << pcl_wait_save->points.size() << RESET << std::endl;
      
      pcd_writer.writeBinary(downsampled_points_dir, *downsampled_cloud); 
      std::cout << GREEN << "Downsampled point cloud data saved to: " << downsampled_points_dir 
                << " with point count after filtering: " << downsampled_cloud->points.size() << RESET << std::endl;

      if(colmap_output_en)
      {
        fout_points << "# 3D point list with one line of data per point\n";
        fout_points << "#  POINT_ID, X, Y, Z, R, G, B, ERROR\n";
        for (size_t i = 0; i < downsampled_cloud->size(); ++i) 
        {
            const auto& point = downsampled_cloud->points[i];
            fout_points << i << " "
                        << std::fixed << std::setprecision(6)
                        << point.x << " " << point.y << " " << point.z << " "
                        << static_cast<int>(point.r) << " "
                        << static_cast<int>(point.g) << " "
                        << static_cast<int>(point.b) << " "
                        << 0 << std::endl;
        }
      }
    }
    else
    {      
      pcd_writer.writeBinary(raw_points_dir, *pcl_wait_save_intensity);
      std::cout << GREEN << "Raw point cloud data saved to: " << raw_points_dir 
                << " with point count: " << pcl_wait_save_intensity->points.size() << RESET << std::endl;
    }
  }
}

void LIVMapper::run() 
{
  rclcpp::Rate rate(5000);
  while (rclcpp::ok()) 
  {
    // ROS2 Equivalent of spinOnce(). 
    // Requires that the external main function does NOT call spin(), just instantiates the class.
    rclcpp::spin_some(this->get_node_base_interface());

    if (!sync_packages(LidarMeasures)) 
    {
      rate.sleep();
      continue;
    }
    handleFirstFrame();

    processImu();

    stateEstimationAndMapping();
  }
  savePCD();
}

void LIVMapper::prop_imu_once(StatesGroup &imu_prop_state, const double dt, V3D acc_avr, V3D angvel_avr)
{
  double mean_acc_norm = p_imu->IMU_mean_acc_norm;
  acc_avr = acc_avr * G_m_s2 / mean_acc_norm - imu_prop_state.bias_a;
  angvel_avr -= imu_prop_state.bias_g;

  M3D Exp_f = Exp(angvel_avr, dt);
  imu_prop_state.rot_end = imu_prop_state.rot_end * Exp_f;
  V3D acc_imu = imu_prop_state.rot_end * acc_avr + V3D(imu_prop_state.gravity[0], imu_prop_state.gravity[1], imu_prop_state.gravity[2]);
  imu_prop_state.pos_end = imu_prop_state.pos_end + imu_prop_state.vel_end * dt + 0.5 * acc_imu * dt * dt;
  imu_prop_state.vel_end = imu_prop_state.vel_end + acc_imu * dt;
}

void LIVMapper::imu_prop_callback()
{
  if (p_imu->imu_need_init || !new_imu || !ekf_finish_once) { return; }
  mtx_buffer_imu_prop.lock();
  new_imu = false; 
  if (imu_prop_enable && !prop_imu_buffer.empty())
  {
    static double last_t_from_lidar_end_time = 0;
    if (state_update_flg)
    {
      imu_propagate = latest_ekf_state;
      // drop all useless imu pkg
      while ((!prop_imu_buffer.empty() && timeToSec(prop_imu_buffer.front().header.stamp) < latest_ekf_time))
      {
        prop_imu_buffer.pop_front();
      }
      last_t_from_lidar_end_time = 0;
      for (int i = 0; i < prop_imu_buffer.size(); i++)
      {
        double t_from_lidar_end_time = timeToSec(prop_imu_buffer[i].header.stamp) - latest_ekf_time;
        double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;
        V3D acc_imu(prop_imu_buffer[i].linear_acceleration.x, prop_imu_buffer[i].linear_acceleration.y, prop_imu_buffer[i].linear_acceleration.z);
        V3D omg_imu(prop_imu_buffer[i].angular_velocity.x, prop_imu_buffer[i].angular_velocity.y, prop_imu_buffer[i].angular_velocity.z);
        prop_imu_once(imu_propagate, dt, acc_imu, omg_imu);
        last_t_from_lidar_end_time = t_from_lidar_end_time;
      }
      state_update_flg = false;
    }
    else
    {
      V3D acc_imu(newest_imu.linear_acceleration.x, newest_imu.linear_acceleration.y, newest_imu.linear_acceleration.z);
      V3D omg_imu(newest_imu.angular_velocity.x, newest_imu.angular_velocity.y, newest_imu.angular_velocity.z);
      double t_from_lidar_end_time = timeToSec(newest_imu.header.stamp) - latest_ekf_time;
      double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;
      prop_imu_once(imu_propagate, dt, acc_imu, omg_imu);
      last_t_from_lidar_end_time = t_from_lidar_end_time;
    }

    V3D posi, vel_i;
    Eigen::Quaterniond q;
    posi = imu_propagate.pos_end;
    vel_i = imu_propagate.vel_end;
    q = Eigen::Quaterniond(imu_propagate.rot_end);
    imu_prop_odom.header.frame_id = "world";
    imu_prop_odom.header.stamp = newest_imu.header.stamp;
    imu_prop_odom.pose.pose.position.x = posi.x();
    imu_prop_odom.pose.pose.position.y = posi.y();
    imu_prop_odom.pose.pose.position.z = posi.z();
    imu_prop_odom.pose.pose.orientation.w = q.w();
    imu_prop_odom.pose.pose.orientation.x = q.x();
    imu_prop_odom.pose.pose.orientation.y = q.y();
    imu_prop_odom.pose.pose.orientation.z = q.z();
    imu_prop_odom.twist.twist.linear.x = vel_i.x();
    imu_prop_odom.twist.twist.linear.y = vel_i.y();
    imu_prop_odom.twist.twist.linear.z = vel_i.z();
    pubImuPropOdom->publish(imu_prop_odom);
  }
  mtx_buffer_imu_prop.unlock();
}

void LIVMapper::transformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud, PointCloudXYZI::Ptr &trans_cloud)
{
  PointCloudXYZI().swap(*trans_cloud);
  trans_cloud->reserve(input_cloud->size());
  for (size_t i = 0; i < input_cloud->size(); i++)
  {
    pcl::PointXYZINormal p_c = input_cloud->points[i];
    Eigen::Vector3d p(p_c.x, p_c.y, p_c.z);
    p = (rot * (extR * p + extT) + t);
    PointType pi;
    pi.x = p(0);
    pi.y = p(1);
    pi.z = p(2);
    pi.intensity = p_c.intensity;
    trans_cloud->points.push_back(pi);
  }
}

void LIVMapper::pointBodyToWorld(const PointType &pi, PointType &po)
{
  V3D p_body(pi.x, pi.y, pi.z);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po.x = p_global(0);
  po.y = p_global(1);
  po.z = p_global(2);
  po.intensity = pi.intensity;
}

template <typename T> void LIVMapper::pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
  V3D p_body(pi[0], pi[1], pi[2]);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po[0] = p_global(0);
  po[1] = p_global(1);
  po[2] = p_global(2);
}

template <typename T> Matrix<T, 3, 1> LIVMapper::pointBodyToWorld(const Matrix<T, 3, 1> &pi)
{
  V3D p(pi[0], pi[1], pi[2]);
  p = (_state.rot_end * (extR * p + extT) + _state.pos_end);
  Matrix<T, 3, 1> po(p[0], p[1], p[2]);
  return po;
}

void LIVMapper::RGBpointBodyToWorld(PointType const *const pi, PointType *const po)
{
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void LIVMapper::standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
  if (!lidar_en) return;
  mtx_buffer.lock();

  double cur_head_time = timeToSec(msg->header.stamp) + lidar_time_offset;
  if (cur_head_time < last_timestamp_lidar)
  {
    RCLCPP_ERROR(this->get_logger(), "lidar loop back, clear buffer");
    lid_raw_data_buffer.clear();
  }
  
  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);
  lid_raw_data_buffer.push_back(ptr);
  lid_header_time_buffer.push_back(cur_head_time);
  last_timestamp_lidar = cur_head_time;

  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

void LIVMapper::livox_pcl_cbk(const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr msg_in)
{
  if (!lidar_en) return;
  mtx_buffer.lock();
  
  auto msg = std::make_shared<livox_ros_driver2::msg::CustomMsg>(*msg_in);

  if (abs(last_timestamp_imu - timeToSec(msg->header.stamp)) > 1.0 && !imu_buffer.empty())
  {
    double timediff_imu_wrt_lidar = last_timestamp_imu - timeToSec(msg->header.stamp);
    printf("\033[95mSelf sync IMU and LiDAR, HARD time lag is %.10lf \n\033[0m", timediff_imu_wrt_lidar - 0.100);
  }

  double cur_head_time = timeToSec(msg->header.stamp);
  RCLCPP_INFO(this->get_logger(), "Get LiDAR, its header time: %.6f", cur_head_time);
  if (cur_head_time < last_timestamp_lidar)
  {
    RCLCPP_ERROR(this->get_logger(), "lidar loop back, clear buffer");
    lid_raw_data_buffer.clear();
  }
  
  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);

  if (!ptr || ptr->empty()) {
    RCLCPP_ERROR(this->get_logger(), "Received an empty point cloud");
    mtx_buffer.unlock();
    return;
  }

  lid_raw_data_buffer.push_back(ptr);
  lid_header_time_buffer.push_back(cur_head_time);
  last_timestamp_lidar = cur_head_time;

  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

void LIVMapper::imu_cbk(const sensor_msgs::msg::Imu::ConstSharedPtr msg_in)
{
  if (!imu_en) return;

  if (last_timestamp_lidar < 0.0) return;
  
  auto msg = std::make_shared<sensor_msgs::msg::Imu>(*msg_in);
  
  double original_ts = timeToSec(msg->header.stamp);
  double timestamp = original_ts - imu_time_offset;
  msg->header.stamp = rclcpp::Time(static_cast<int64_t>(timestamp * 1e9));

  if (fabs(last_timestamp_lidar - timestamp) > 0.5 && (!ros_driver_fix_en))
  {
    RCLCPP_WARN(this->get_logger(), "IMU and LiDAR not synced! delta time: %lf .", last_timestamp_lidar - timestamp);
  }

  if (ros_driver_fix_en) {
      timestamp += std::round(last_timestamp_lidar - timestamp);
      msg->header.stamp = rclcpp::Time(static_cast<int64_t>(timestamp * 1e9));
  }

  mtx_buffer.lock();

  if (last_timestamp_imu > 0.0 && timestamp < last_timestamp_imu)
  {
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    RCLCPP_ERROR(this->get_logger(), "imu loop back, offset: %lf", last_timestamp_imu - timestamp);
    return;
  }

  last_timestamp_imu = timestamp;

  imu_buffer.push_back(msg);
  mtx_buffer.unlock();
  if (imu_prop_enable)
  {
    mtx_buffer_imu_prop.lock();
    if (imu_prop_enable && !p_imu->imu_need_init) { prop_imu_buffer.push_back(*msg); }
    newest_imu = *msg;
    new_imu = true;
    mtx_buffer_imu_prop.unlock();
  }
  sig_buffer.notify_all();
}

cv::Mat LIVMapper::getImageFromMsg(const sensor_msgs::msg::Image::ConstSharedPtr &img_msg)
{
  cv::Mat img;
  try {
      img = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
  } catch (cv_bridge::Exception& e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
  }
  return img;
}

void LIVMapper::img_cbk(const sensor_msgs::msg::Image::ConstSharedPtr msg_in)
{
  if (!img_en) return;
  auto msg = std::make_shared<sensor_msgs::msg::Image>(*msg_in);
 
  if (hilti_en)
  {
    static int frame_counter = 0;
    if (++frame_counter % 4 != 0) return;
  }
  
  double msg_header_time = timeToSec(msg->header.stamp) + img_time_offset;
  if (abs(msg_header_time - last_timestamp_img) < 0.001) return;
  RCLCPP_INFO(this->get_logger(), "Get image, its header time: %.6f", msg_header_time);
  if (last_timestamp_lidar < 0) return;

  if (msg_header_time < last_timestamp_img)
  {
    RCLCPP_ERROR(this->get_logger(), "image loop back.");
    return;
  }

  mtx_buffer.lock();

  double img_time_correct = msg_header_time;

  if (img_time_correct - last_timestamp_img < 0.02)
  {
    RCLCPP_WARN(this->get_logger(), "Image need Jumps: %.6f", img_time_correct);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    return;
  }

  cv::Mat img_cur = getImageFromMsg(msg);
  if(img_cur.empty()) {
     mtx_buffer.unlock();
     return;
  }
  
  img_buffer.push_back(img_cur);
  img_time_buffer.push_back(img_time_correct);

  last_timestamp_img = img_time_correct;
  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

bool LIVMapper::sync_packages(LidarMeasureGroup &meas)
{
  if (lid_raw_data_buffer.empty() && lidar_en) return false;
  if (img_buffer.empty() && img_en) return false;
  if (imu_buffer.empty() && imu_en) return false;

  switch (slam_mode_)
  {
  case ONLY_LIO:
  {
    if (meas.last_lio_update_time < 0.0) meas.last_lio_update_time = lid_header_time_buffer.front();
    if (!lidar_pushed)
    {
      meas.lidar = lid_raw_data_buffer.front(); 
      if (meas.lidar->points.size() <= 1) return false;

      meas.lidar_frame_beg_time = lid_header_time_buffer.front();                                                
      meas.lidar_frame_end_time = meas.lidar_frame_beg_time + meas.lidar->points.back().curvature / double(1000); 
      meas.pcl_proc_cur = meas.lidar;
      lidar_pushed = true;                                                                                       
    }

    if (imu_en && last_timestamp_imu < meas.lidar_frame_end_time)
    { 
      return false;
    }

    struct MeasureGroup m; 

    m.imu.clear();
    m.lio_time = meas.lidar_frame_end_time;
    mtx_buffer.lock();
    while (!imu_buffer.empty())
    {
      if (timeToSec(imu_buffer.front()->header.stamp) > meas.lidar_frame_end_time) break;
      m.imu.push_back(imu_buffer.front());
      imu_buffer.pop_front();
    }
    lid_raw_data_buffer.pop_front();
    lid_header_time_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();

    meas.lio_vio_flg = LIO; 
    meas.measures.push_back(m);
    lidar_pushed = false; 
    return true;

    break;
  }

  case LIVO:
  {
    EKF_STATE last_lio_vio_flg = meas.lio_vio_flg;
    switch (last_lio_vio_flg)
    {
    case WAIT:
    case VIO:
    {
      double img_capture_time = img_time_buffer.front() + exposure_time_init;
      if (meas.last_lio_update_time < 0.0) meas.last_lio_update_time = lid_header_time_buffer.front();

      double lid_newest_time = lid_header_time_buffer.back() + lid_raw_data_buffer.back()->points.back().curvature / double(1000);
      double imu_newest_time = timeToSec(imu_buffer.back()->header.stamp);

      if (img_capture_time < meas.last_lio_update_time + 0.00001)
      {
        img_buffer.pop_front();
        img_time_buffer.pop_front();
        RCLCPP_ERROR(this->get_logger(), "[ Data Cut ] Throw one image frame!");
        return false;
      }

      if (img_capture_time > lid_newest_time || img_capture_time > imu_newest_time)
      {
        return false;
      }

      struct MeasureGroup m;

      m.imu.clear();
      m.lio_time = img_capture_time;
      mtx_buffer.lock();
      while (!imu_buffer.empty())
      {
        if (timeToSec(imu_buffer.front()->header.stamp) > m.lio_time) break;

        if (timeToSec(imu_buffer.front()->header.stamp) > meas.last_lio_update_time) m.imu.push_back(imu_buffer.front());

        imu_buffer.pop_front();
      }
      mtx_buffer.unlock();
      sig_buffer.notify_all();

      *(meas.pcl_proc_cur) = *(meas.pcl_proc_next);
      PointCloudXYZI().swap(*meas.pcl_proc_next);

      int lid_frame_num = lid_raw_data_buffer.size();
      int max_size = meas.pcl_proc_cur->size() + 24000 * lid_frame_num;
      meas.pcl_proc_cur->reserve(max_size);
      meas.pcl_proc_next->reserve(max_size);

      while (!lid_raw_data_buffer.empty())
      {
        if (lid_header_time_buffer.front() > img_capture_time) break;
        auto pcl(lid_raw_data_buffer.front()->points);
        double frame_header_time(lid_header_time_buffer.front());
        float max_offs_time_ms = (m.lio_time - frame_header_time) * 1000.0f;

        for (int i = 0; i < pcl.size(); i++)
        {
          auto pt = pcl[i];
          if (pcl[i].curvature < max_offs_time_ms)
          {
            pt.curvature += (frame_header_time - meas.last_lio_update_time) * 1000.0f;
            meas.pcl_proc_cur->points.push_back(pt);
          }
          else
          {
            pt.curvature += (frame_header_time - m.lio_time) * 1000.0f;
            meas.pcl_proc_next->points.push_back(pt);
          }
        }
        lid_raw_data_buffer.pop_front();
        lid_header_time_buffer.pop_front();
      }

      meas.measures.push_back(m);
      meas.lio_vio_flg = LIO;
      return true;
    }

    case LIO:
    {
      double img_capture_time = img_time_buffer.front() + exposure_time_init;
      meas.lio_vio_flg = VIO;
      meas.measures.clear();
      // double imu_time = timeToSec(imu_buffer.front()->header.stamp);

      struct MeasureGroup m;
      m.vio_time = img_capture_time;
      m.lio_time = meas.last_lio_update_time;
      m.img = img_buffer.front();
      mtx_buffer.lock();
      img_buffer.pop_front();
      img_time_buffer.pop_front();
      mtx_buffer.unlock();
      sig_buffer.notify_all();
      meas.measures.push_back(m);
      lidar_pushed = false; 
      return true;
    }

    default:
    {
      return false;
    }
    }
    break;
  }

  case ONLY_LO:
  {
    if (!lidar_pushed) 
    { 
      if (lid_raw_data_buffer.empty())  return false;
      meas.lidar = lid_raw_data_buffer.front(); 
      meas.lidar_frame_beg_time = lid_header_time_buffer.front(); 
      meas.lidar_frame_end_time  = meas.lidar_frame_beg_time + meas.lidar->points.back().curvature / double(1000); 
      lidar_pushed = true;             
    }
    struct MeasureGroup m; 
    m.lio_time = meas.lidar_frame_end_time;
    mtx_buffer.lock();
    lid_raw_data_buffer.pop_front();
    lid_header_time_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    lidar_pushed = false; 
    meas.lio_vio_flg = LO; 
    meas.measures.push_back(m);
    return true;
    break;
  }

  default:
  {
    printf("!! WRONG SLAM TYPE !!");
    return false;
  }
  }
  RCLCPP_ERROR(this->get_logger(), "out sync");
  return false;
}

void LIVMapper::publish_img_rgb(const image_transport::Publisher &pubImage, VIOManagerPtr vio_manager)
{
  cv::Mat img_rgb = vio_manager->img_cp;
  cv_bridge::CvImage out_msg;
  out_msg.header.stamp = this->now();
  out_msg.encoding = sensor_msgs::image_encodings::BGR8;
  out_msg.image = img_rgb;
  pubImage.publish(out_msg.toImageMsg());
}

void LIVMapper::publish_frame_world(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pubLaserCloudFullRes, VIOManagerPtr vio_manager)
{
  if (pcl_w_wait_pub->empty()) return;
  PointCloudXYZRGB::Ptr laserCloudWorldRGB(new PointCloudXYZRGB());
  if (img_en)
  {
    static int pub_num = 1;
    *pcl_wait_pub += *pcl_w_wait_pub;
    if(pub_num == pub_scan_num)
    {
      pub_num = 1;
      size_t size = pcl_wait_pub->points.size();
      laserCloudWorldRGB->reserve(size);
      cv::Mat img_rgb = vio_manager->img_rgb;
      for (size_t i = 0; i < size; i++)
      {
        PointTypeRGB pointRGB;
        pointRGB.x = pcl_wait_pub->points[i].x;
        pointRGB.y = pcl_wait_pub->points[i].y;
        pointRGB.z = pcl_wait_pub->points[i].z;

        V3D p_w(pcl_wait_pub->points[i].x, pcl_wait_pub->points[i].y, pcl_wait_pub->points[i].z);
        V3D pf(vio_manager->new_frame_->w2f(p_w)); if (pf[2] < 0) continue;
        V2D pc(vio_manager->new_frame_->w2c(p_w));

        if (vio_manager->new_frame_->cam_->isInFrame(pc.cast<int>(), 3)) 
        {
          V3F pixel = vio_manager->getInterpolatedPixel(img_rgb, pc);
          pointRGB.r = pixel[2];
          pointRGB.g = pixel[1];
          pointRGB.b = pixel[0];
          if (pf.norm() > blind_rgb_points) laserCloudWorldRGB->push_back(pointRGB);
        }
      }
    }
    else
    {
      pub_num++;
    }
  }

  sensor_msgs::msg::PointCloud2 laserCloudmsg;
  if (img_en)
  {
    pcl::toROSMsg(*laserCloudWorldRGB, laserCloudmsg);
  }
  else 
  { 
    pcl::toROSMsg(*pcl_w_wait_pub, laserCloudmsg); 
  }
  laserCloudmsg.header.stamp = this->now();
  laserCloudmsg.header.frame_id = "camera_init";
  pubLaserCloudFullRes->publish(laserCloudmsg);

  if (pcd_save_en)
  {
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));
    static int scan_wait_num = 0;

    if (img_en)
    {
      *pcl_wait_save += *laserCloudWorldRGB;
    }
    else
    {
      *pcl_wait_save_intensity += *pcl_w_wait_pub;
    }
    scan_wait_num++;

    if ((pcl_wait_save->size() > 0 || pcl_wait_save_intensity->size() > 0) && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval)
    {
      pcd_index++;
      string all_points_dir(string(string(ROOT_DIR) + "Log/PCD/") + to_string(pcd_index) + string(".pcd"));
      pcl::PCDWriter pcd_writer;
      if (pcd_save_en)
      {
        cout << "current scan saved to /PCD/" << all_points_dir << endl;
        if (img_en)
        {
          pcd_writer.writeBinary(all_points_dir, *pcl_wait_save); 
          PointCloudXYZRGB().swap(*pcl_wait_save);
        }
        else
        {
          pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_intensity);
          PointCloudXYZI().swap(*pcl_wait_save_intensity);
        }        
        Eigen::Quaterniond q(_state.rot_end);
        fout_pcd_pos << _state.pos_end[0] << " " << _state.pos_end[1] << " " << _state.pos_end[2] << " " << q.w() << " " << q.x() << " " << q.y()
                     << " " << q.z() << " " << endl;
        scan_wait_num = 0;
      }
    }
  }
  if(laserCloudWorldRGB->size() > 0)  PointCloudXYZI().swap(*pcl_wait_pub); 
  PointCloudXYZI().swap(*pcl_w_wait_pub);
}

void LIVMapper::publish_visual_sub_map(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pubSubVisualMap)
{
  PointCloudXYZI::Ptr laserCloudFullRes(visual_sub_map);
  int size = laserCloudFullRes->points.size(); if (size == 0) return;
  PointCloudXYZI::Ptr sub_pcl_visual_map_pub(new PointCloudXYZI());
  *sub_pcl_visual_map_pub = *laserCloudFullRes;
  if (1)
  {
    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*sub_pcl_visual_map_pub, laserCloudmsg);
    laserCloudmsg.header.stamp = this->now();
    laserCloudmsg.header.frame_id = "camera_init";
    pubSubVisualMap->publish(laserCloudmsg);
  }
}

void LIVMapper::publish_effect_world(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pubLaserCloudEffect, const std::vector<PointToPlane> &ptpl_list)
{
  int effect_feat_num = ptpl_list.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(effect_feat_num, 1));
  for (int i = 0; i < effect_feat_num; i++)
  {
    laserCloudWorld->points[i].x = ptpl_list[i].point_w_[0];
    laserCloudWorld->points[i].y = ptpl_list[i].point_w_[1];
    laserCloudWorld->points[i].z = ptpl_list[i].point_w_[2];
  }
  sensor_msgs::msg::PointCloud2 laserCloudFullRes3;
  pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
  laserCloudFullRes3.header.stamp = this->now();
  laserCloudFullRes3.header.frame_id = "camera_init";
  pubLaserCloudEffect->publish(laserCloudFullRes3);
}

template <typename T> void LIVMapper::set_posestamp(T &out)
{
  out.position.x = _state.pos_end(0);
  out.position.y = _state.pos_end(1);
  out.position.z = _state.pos_end(2);
  out.orientation.x = geoQuat.x;
  out.orientation.y = geoQuat.y;
  out.orientation.z = geoQuat.z;
  out.orientation.w = geoQuat.w;
}

void LIVMapper::publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr &pubOdomAftMapped)
{
  odomAftMapped.header.frame_id = "camera_init";
  odomAftMapped.child_frame_id = "aft_mapped";
  odomAftMapped.header.stamp = rclcpp::Time(static_cast<int64_t>(LidarMeasures.last_lio_update_time * 1e9));
  set_posestamp(odomAftMapped.pose.pose);

  geometry_msgs::msg::TransformStamped transform;
  transform.header.stamp = odomAftMapped.header.stamp;
  transform.header.frame_id = "camera_init";
  transform.child_frame_id = "aft_mapped";
  transform.transform.translation.x = _state.pos_end(0);
  transform.transform.translation.y = _state.pos_end(1);
  transform.transform.translation.z = _state.pos_end(2);
  transform.transform.rotation.w = geoQuat.w;
  transform.transform.rotation.x = geoQuat.x;
  transform.transform.rotation.y = geoQuat.y;
  transform.transform.rotation.z = geoQuat.z;
  
  tf_broadcaster_->sendTransform(transform);
  pubOdomAftMapped->publish(odomAftMapped);
}

void LIVMapper::publish_odometry_cam(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr &pubOdomAftMapped)
{
  M3D R_w_c = _state.rot_end * vio_manager->Rci.transpose();
  V3D t_w_c = _state.pos_end - _state.rot_end * vio_manager->Rci.transpose() * vio_manager->Pci;
  
  Eigen::Quaterniond q_w_c(R_w_c);
  
  nav_msgs::msg::Odometry camOdomAftMapped;
  camOdomAftMapped.header.frame_id = "camera_init";
  camOdomAftMapped.child_frame_id = "camera";
  camOdomAftMapped.header.stamp = rclcpp::Time(static_cast<int64_t>(LidarMeasures.last_lio_update_time * 1e9));
  camOdomAftMapped.pose.pose.position.x = t_w_c(0);
  camOdomAftMapped.pose.pose.position.y = t_w_c(1);
  camOdomAftMapped.pose.pose.position.z = t_w_c(2);
  camOdomAftMapped.pose.pose.orientation.x = q_w_c.x();
  camOdomAftMapped.pose.pose.orientation.y = q_w_c.y();
  camOdomAftMapped.pose.pose.orientation.z = q_w_c.z();
  camOdomAftMapped.pose.pose.orientation.w = q_w_c.w();

  geometry_msgs::msg::TransformStamped transform;
  transform.header.stamp = camOdomAftMapped.header.stamp;
  transform.header.frame_id = "camera_init";
  transform.child_frame_id = "camera";
  transform.transform.translation.x = t_w_c(0);
  transform.transform.translation.y = t_w_c(1);
  transform.transform.translation.z = t_w_c(2);
  transform.transform.rotation.w = q_w_c.w();
  transform.transform.rotation.x = q_w_c.x();
  transform.transform.rotation.y = q_w_c.y();
  transform.transform.rotation.z = q_w_c.z();

  tf_broadcaster_->sendTransform(transform);
  pubOdomAftMapped->publish(camOdomAftMapped);
}

void LIVMapper::publish_odometry_lidar(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr &pubOdomAftMapped)
{
  M3D R_w_l = _state.rot_end * vio_manager->Rli;
  V3D t_w_l = _state.pos_end + _state.rot_end * vio_manager->Pli;
  
  Eigen::Quaterniond q_w_l(R_w_l);
  
  nav_msgs::msg::Odometry lidarOdomAftMapped;
  lidarOdomAftMapped.header.frame_id = "camera_init";
  lidarOdomAftMapped.child_frame_id = "lidar";
  lidarOdomAftMapped.header.stamp = rclcpp::Time(static_cast<int64_t>(LidarMeasures.last_lio_update_time * 1e9));
  lidarOdomAftMapped.pose.pose.position.x = t_w_l(0);
  lidarOdomAftMapped.pose.pose.position.y = t_w_l(1);
  lidarOdomAftMapped.pose.pose.position.z = t_w_l(2);
  lidarOdomAftMapped.pose.pose.orientation.x = q_w_l.x();
  lidarOdomAftMapped.pose.pose.orientation.y = q_w_l.y();
  lidarOdomAftMapped.pose.pose.orientation.z = q_w_l.z();
  lidarOdomAftMapped.pose.pose.orientation.w = q_w_l.w();

  geometry_msgs::msg::TransformStamped transform;
  transform.header.stamp = lidarOdomAftMapped.header.stamp;
  transform.header.frame_id = "camera_init";
  transform.child_frame_id = "lidar";
  transform.transform.translation.x = t_w_l(0);
  transform.transform.translation.y = t_w_l(1);
  transform.transform.translation.z = t_w_l(2);
  transform.transform.rotation.w = q_w_l.w();
  transform.transform.rotation.x = q_w_l.x();
  transform.transform.rotation.y = q_w_l.y();
  transform.transform.rotation.z = q_w_l.z();

  tf_broadcaster_->sendTransform(transform);
  pubOdomAftMapped->publish(lidarOdomAftMapped);
}

void LIVMapper::publish_mavros(const rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr &mavros_pose_publisher)
{
  msg_body_pose.header.stamp = this->now();
  msg_body_pose.header.frame_id = "camera_init";
  set_posestamp(msg_body_pose.pose);
  mavros_pose_publisher->publish(msg_body_pose);
}

void LIVMapper::publish_path(const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath)
{
  set_posestamp(msg_body_pose.pose);
  msg_body_pose.header.stamp = this->now();
  msg_body_pose.header.frame_id = "camera_init";
  path.poses.push_back(msg_body_pose);
  pubPath->publish(path);
}
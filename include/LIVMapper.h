#ifndef LIV_MAPPER_H
#define LIV_MAPPER_H

// ROS 2 Core
#include <rclcpp/rclcpp.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.hpp>

// ROS 2 Messages
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

// TF2
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// STD / PCL / Custom
#include <mutex>
#include <condition_variable>
#include <deque>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <memory>

#include "IMU_Processing.h"
#include "vio.h"
#include "preprocess.h"
#include <vikit/camera_loader.h>

class LIVMapper : public rclcpp::Node
{
public:
  explicit LIVMapper(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~LIVMapper();

  /* ---------------- Initialization ---------------- */
  void initializeSubscribersAndPublishers();
  void initializeComponents();
  void initializeFiles();
  void readParameters();

  /* ---------------- Main Pipeline ---------------- */
  void run();
  void gravityAlignment();
  void handleFirstFrame();
  void stateEstimationAndMapping();
  void handleVIO();
  void handleLIO();
  void savePCD();
  void processImu();

  /* ---------------- IMU Logic ---------------- */
  bool sync_packages(LidarMeasureGroup &meas);
  void prop_imu_once(StatesGroup &imu_prop_state, const double dt, V3D acc_avr, V3D angvel_avr);
  void imu_prop_callback();

  /* ---------------- Point Cloud Transforms ---------------- */
  void transformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud, PointCloudXYZI::Ptr &trans_cloud);
  void pointBodyToWorld(const PointType &pi, PointType &po);
  void RGBpointBodyLidarToIMU(PointType const *const pi, PointType *const po);
  void RGBpointBodyToWorld(PointType const *const pi, PointType *const po);

  /* ---------------- Callbacks ---------------- */
  void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
  void livox_pcl_cbk(const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr msg);
  void imu_cbk(const sensor_msgs::msg::Imu::ConstSharedPtr msg);
  void img_cbk(const sensor_msgs::msg::Image::ConstSharedPtr msg);

  /* ---------------- Publishing ---------------- */
  void publish_img_rgb(const image_transport::Publisher &pubImage, VIOManagerPtr vio_manager);
  void publish_frame_world(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pubLaserCloudFullRes, VIOManagerPtr vio_manager);
  void publish_visual_sub_map(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pubSubVisualMap);
  void publish_effect_world(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pubLaserCloudEffect, const std::vector<PointToPlane> &ptpl_list);
  void publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr &pubOdomAftMapped);
  void publish_odometry_cam(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr &pubOdomAftMapped);
  void publish_odometry_lidar(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr &pubOdomAftMapped);
  void publish_mavros(const rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr &mavros_pose_publisher);
  void publish_path(const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath);

  /* ---------------- Utils ---------------- */
  template <typename T> void set_posestamp(T &out);
  template <typename T> void pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi, Eigen::Matrix<T, 3, 1> &po);
  template <typename T> Eigen::Matrix<T, 3, 1> pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi);
  cv::Mat getImageFromMsg(const sensor_msgs::msg::Image::ConstSharedPtr &img_msg);

  /* ---------------- Threading & Synchronization ---------------- */
  std::mutex mtx_buffer, mtx_buffer_imu_prop;
  std::condition_variable sig_buffer;

  /* ---------------- State & Map ---------------- */
  SLAM_MODE slam_mode_;
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> voxel_map;

  std::string root_dir;
  std::string lid_topic, imu_topic, seq_name, img_topic;
  V3D extT;
  M3D extR;

  /* ---------------- Parameters ---------------- */
  int feats_down_size = 0;
  int max_iterations = 0;

  double res_mean_last = 0.05;
  double gyr_cov = 0, acc_cov = 0, inv_expo_cov = 0;
  double blind_rgb_points = 0.0;
  double last_timestamp_lidar = -1.0, last_timestamp_imu = -1.0, last_timestamp_img = -1.0;
  double filter_size_surf_min = 0;
  double filter_size_pcd = 0;
  double _first_lidar_time = 0.0;
  double match_time = 0, solve_time = 0, solve_const_H_time = 0;

  bool lidar_map_inited = false, pcd_save_en = false, img_save_en = false, pub_effect_point_en = false, pose_output_en = false, ros_driver_fix_en = false, hilti_en = false, pub_tf_odom = false;
  int img_save_interval = 1, pcd_save_interval = -1, pcd_save_type = 0;
  int pub_scan_num = 1;

  /* ---------------- IMU Propagation ---------------- */
  StatesGroup imu_propagate, latest_ekf_state;
  bool new_imu = false, state_update_flg = false, imu_prop_enable = true, ekf_finish_once = false;
  
  std::deque<sensor_msgs::msg::Imu> prop_imu_buffer;
  sensor_msgs::msg::Imu newest_imu;
  
  double latest_ekf_time;
  nav_msgs::msg::Odometry imu_prop_odom;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubImuPropOdom;
  
  double imu_time_offset = 0.0;
  double lidar_time_offset = 0.0;

  bool gravity_align_en = false, gravity_align_finished = false;
  bool sync_jump_flag = false;
  bool lidar_pushed = false, imu_en, gravity_est_en, flg_reset = false, ba_bg_est_en = true;
  bool dense_map_en = false;
  int img_en = 1, imu_int_frame = 3;
  bool normal_en = true;
  bool exposure_estimate_en = false;
  double exposure_time_init = 0.0;
  bool inverse_composition_en = false;
  bool raycast_en = false;
  int lidar_en = 1;
  bool is_first_frame = false;
  int grid_size, patch_size, grid_n_width, grid_n_height, patch_pyrimid_level;
  double outlier_threshold;
  double plot_time;
  int frame_cnt;
  double img_time_offset = 0.0;
  int pcd_index = 0;

  /* ---------------- Buffers ---------------- */
  std::deque<PointCloudXYZI::Ptr> lid_raw_data_buffer;
  std::deque<double> lid_header_time_buffer;
  std::deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer;
  std::deque<cv::Mat> img_buffer;
  std::deque<double> img_time_buffer;
  
  std::vector<pointWithVar> _pv_list;
  std::vector<double> extrinT;
  std::vector<double> extrinR;
  std::vector<double> cameraextrinT;
  std::vector<double> cameraextrinR;
  double IMG_POINT_COV;

  /* ---------------- Point Cloud Objects ---------------- */
  PointCloudXYZI::Ptr visual_sub_map;
  PointCloudXYZI::Ptr feats_undistort;
  PointCloudXYZI::Ptr feats_down_body;
  PointCloudXYZI::Ptr feats_down_world;
  PointCloudXYZI::Ptr pcl_w_wait_pub;
  PointCloudXYZI::Ptr pcl_l_wait_pub;
  PointCloudXYZI::Ptr pcl_wait_pub;
  PointCloudXYZRGB::Ptr pcl_wait_save;
  PointCloudXYZI::Ptr pcl_wait_save_intensity;

  /* ---------------- File IO ---------------- */
  std::ofstream fout_pre, fout_out, fout_visual_pos, fout_pcd_pos, fout_lidar_pos, fout_points;

  pcl::VoxelGrid<PointType> downSizeFilterSurf;

  V3D euler_cur;

  /* ---------------- Measurements & States ---------------- */
  LidarMeasureGroup LidarMeasures;
  StatesGroup _state;
  StatesGroup state_propagat;

  /* ---------------- ROS Messages ---------------- */
  nav_msgs::msg::Path path;
  nav_msgs::msg::Odometry odomAftMapped;
  geometry_msgs::msg::Quaternion geoQuat;
  geometry_msgs::msg::PoseStamped msg_body_pose;

  /* ---------------- Module Pointers ---------------- */
  PreprocessPtr p_pre;
  ImuProcessPtr p_imu;
  VoxelMapManagerPtr voxelmap_manager;
  VIOManagerPtr vio_manager;

  /* ---------------- ROS 2 Publishers & Subscribers ---------------- */
  rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl_livox;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_std;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img;

  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr plane_pub;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr voxel_pub;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pubNormal;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFullRes;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFullResBody;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubSubVisualMap;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap;
  
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMappedCam;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMappedLiDAR;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath;
  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudDyn;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudDynRmed;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudDynDbg;
  
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr mavros_pose_publisher;
  
  image_transport::Publisher pubImage;
  image_transport::Publisher pubOriginImage;

  rclcpp::TimerBase::SharedPtr imu_prop_timer;
  
  // TF Broadcaster
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  int frame_num = 0;
  double aver_time_consu = 0;
  double aver_time_icp = 0;
  double aver_time_map_inre = 0;
  bool colmap_output_en = false;
};

#endif // LIV_MAPPER_H
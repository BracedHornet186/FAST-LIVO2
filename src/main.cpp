#include "LIVMapper.h"

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  auto mapper = std::make_shared<LIVMapper>(options);
  mapper->run();
  rclcpp::shutdown();
  return 0;
}
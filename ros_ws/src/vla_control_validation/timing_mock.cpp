// Copyright 2026 VLA contributors. SPDX-License-Identifier: Apache-2.0
#include <mock_components/generic_system.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <chrono>
#include <fstream>
#include <vector>
#include <cmath>

namespace vla_control_validation {
class TimingMockSystem : public mock_components::GenericSystem {
 public:
  CallbackReturn on_init(const hardware_interface::HardwareInfo& info) override {
    const auto result = GenericSystem::on_init(info);
    if (result != CallbackReturn::SUCCESS) return result;
    path_ = info.hardware_parameters.at("timing_output");
    if (path_.rfind("/tmp/", 0) != 0) return CallbackReturn::ERROR;
    ticks_.reserve(1300000);  // >20 minutes at 1 kHz; never allocate in write().
    previous_.assign(info.joints.size(), 0.0);
    return CallbackReturn::SUCCESS;
  }
  return_type read(const rclcpp::Time&, const rclcpp::Duration&) override {
    return return_type::OK;  // Deliberately no physical dynamics or device I/O.
  }
  return_type write(const rclcpp::Time&, const rclcpp::Duration&) override {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    if (ticks_.size() < ticks_.capacity())
      ticks_.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
    else ++dropped_;
    for (size_t i = 0; i < previous_.size(); ++i) {
      const auto tau = joint_commands_[3][i];
      if (std::isfinite(tau)) {
        max_step_ = std::max(max_step_, std::abs(tau-previous_[i]));
        previous_[i] = tau;
      }
    }
    return return_type::OK;
  }
  ~TimingMockSystem() override {
    if (path_.empty()) return;
    std::ofstream out(path_, std::ios::binary);
    out.write(reinterpret_cast<const char*>(ticks_.data()), ticks_.size()*sizeof(int64_t));
    std::ofstream meta(path_+".json");
    meta << "{\"samples\":" << ticks_.size() << ",\"dropped\":" << dropped_
         << ",\"max_torque_step\":" << max_step_ << "}";
  }
 private:
  std::string path_;
  std::vector<int64_t> ticks_;
  std::vector<double> previous_;
  size_t dropped_{0};
  double max_step_{0};
};
}
PLUGINLIB_EXPORT_CLASS(vla_control_validation::TimingMockSystem, hardware_interface::SystemInterface)

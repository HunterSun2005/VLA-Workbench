// Copyright (c) 2025 Franka Robotics GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <Eigen/Eigen>
#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <string>
#include <atomic>
#include <realtime_tools/realtime_buffer.h>
#include <std_msgs/msg/u_int64.hpp>
#include "franka_fr3_arm_controllers/motion_generator.hpp"

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace franka_fr3_arm_controllers {

/**
 * Controller to move the robot to a desired joint position.
 */
class JointImpedanceController : public controller_interface::ControllerInterface {
 public:
  using Vector7d = Eigen::Matrix<double, 7, 1>;
  [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration()
      const override;
  [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration()
      const override;
  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;
  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;

 private:
  struct CommandPacket {
    std::array<double, 7> positions{};
    uint64_t generation{0};
    uint64_t command_id{0};
    uint64_t sequence{0};
    int64_t stamp_ns{0};
  };
  realtime_tools::RealtimeBuffer<CommandPacket> command_buffer_;
  std::atomic<uint64_t> requested_generation_{0};
  uint64_t active_generation_{0};
  uint64_t received_command_id_{0};
  uint64_t received_sequence_{0};
  bool command_watchdog_fault_{false};
  std::atomic<uint64_t> diagnostic_sequence_{0};
  std::atomic<bool> diagnostic_fault_{false};
  std::array<std::atomic<double>, 7> diagnostic_positions_{};
  std::atomic<uint64_t> diagnostic_generation_{0};
  std::atomic<uint64_t> diagnostic_command_id_{0};
  rclcpp::Subscription<std_msgs::msg::UInt64>::SharedPtr reset_subscriber_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr diagnostic_publisher_;
  rclcpp::TimerBase::SharedPtr diagnostic_timer_;
  std::string arm_id_;
  std::string namespace_prefix_;
  std::string robot_description_;
  const int num_joints = 7;
  Vector7d q_;
  Vector7d dq_;
  Vector7d dq_filtered_;
  Vector7d last_commanded_tau_;
  Vector7d last_commanded_q_goal_;
  Vector7d k_gains_;
  Vector7d d_gains_;
  double k_alpha_;
  double initial_alignment_speed_factor_;
  double max_gello_velocity_;
  double max_delta_tau_per_ms_;
  bool last_commanded_q_goal_initialized_{false};
  bool move_to_start_position_finished_{false};
  bool motion_generator_initialized_{false};
  rclcpp::Time start_time_;
  std::unique_ptr<MotionGenerator> motion_generator_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscriber_ = nullptr;
  bool gello_position_values_valid_ = false;
  std::array<double, 7> gello_position_values_{0, 0, 0, 0, 0, 0, 0};
  bool filtered_gello_position_initialized_{false};
  Vector7d filtered_gello_position_values_;
  rclcpp::Time last_joint_state_time_;
  double gello_deadband_;
  double gello_filter_alpha_;

  Vector7d calculateTauDGains_(const Vector7d& q_goal);
  Vector7d limitGoalVelocity_(const Vector7d& q_goal, double period_seconds);
  Vector7d limitTorqueRate_(const Vector7d& tau_d_calculated, double period_seconds);
  Vector7d filterGelloPositions_(const Vector7d& raw_positions);
  bool validateGains_(const std::vector<double>& gains, const std::string& gains_name);
  bool initializeMotionGenerator_();
  void updateJointStates_();
  void validateGelloPositions_(const sensor_msgs::msg::JointState& msg);
  void jointStateCallback_(const sensor_msgs::msg::JointState msg);
};

}  // namespace franka_fr3_arm_controllers

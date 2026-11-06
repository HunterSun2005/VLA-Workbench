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

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

// Manual transport fixture: valid managed commands, no periodic hidden publisher.
class MockGelloJointStatePublisher : public rclcpp::Node {
 public:
  MockGelloJointStatePublisher() : Node("joint_state_publisher_node") {
    publisher_ = create_publisher<sensor_msgs::msg::JointState>("/gello/joint_states", 3);
  }
  void send(uint64_t generation, uint64_t id, uint64_t sequence,
            const std::vector<double>& q, double age = 0.0) {
    sensor_msgs::msg::JointState msg;
    msg.header.stamp = now() - rclcpp::Duration::from_seconds(age);
    msg.header.frame_id = "vla:"+std::to_string(generation)+":"+std::to_string(id)+":"+std::to_string(sequence);
    for (int i=1; i<=7; ++i) msg.name.push_back("fr3_joint"+std::to_string(i));
    msg.position = q;
    publisher_->publish(msg);
  }
 private:
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr publisher_;
};

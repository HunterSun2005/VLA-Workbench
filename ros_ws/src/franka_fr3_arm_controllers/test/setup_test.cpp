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

#include "include/test_joint_impedance_controller.hpp"
#include <limits>


TEST_F(JointImpedanceControllerTest, ManagedCommandsResetWatchdogAndSlewLimit) {
  ASSERT_EQ(startController(), CallbackReturn::SUCCESS);
  pump(50);
  resetGeneration(10);
  ASSERT_EQ(diagnostic().header.frame_id, "vla:10:0:0");
  auto q = position_states_; for (auto& v : q) v += .03;
  mock_joint_state_publisher_->send(10, 1, 1, q);
  pump(50);
  ASSERT_EQ(diagnostic().header.frame_id, "vla:10:1:0");
  auto accepted = diagnostic().position;
  q[0] += .1;
  mock_joint_state_publisher_->send(10, 1, 1, q);  // duplicate packet
  pump(30);
  EXPECT_EQ(diagnostic().position, accepted);
  for (double age : {1.0, -1.0}) {
    mock_joint_state_publisher_->send(10, 2, 2, q, age);
    pump(30);
    EXPECT_EQ(diagnostic().header.frame_id, "vla:10:1:0");
  }
  mock_joint_state_publisher_->send(9, 99, 99, q);  // revoked generation
  pump(30);
  EXPECT_EQ(diagnostic().header.frame_id, "vla:10:1:0");
  for (auto invalid : {std::vector<double>(6, 0.0),
                       std::vector<double>(7, std::numeric_limits<double>::quiet_NaN())}) {
    mock_joint_state_publisher_->send(10, 2, 2, invalid);
    pump(30);
    EXPECT_EQ(diagnostic().header.frame_id, "vla:10:1:0");
  }
  for (int i=0; i<550; ++i) {
    auto previous = joint_commands_; pump(1);
    for (size_t j=0; j<previous.size(); ++j)
      EXPECT_LE(std::abs(joint_commands_[j]-previous[j]), 1.0+1e-9);
  }
  EXPECT_EQ(diagnostic().header.frame_id, "vla:10:1:1");
  mock_joint_state_publisher_->send(10, 3, 3, q); pump(30);
  EXPECT_EQ(diagnostic().header.frame_id, "vla:10:1:1"); // fault latches
  resetGeneration(11);
  EXPECT_EQ(diagnostic().header.frame_id, "vla:11:0:0");
  mock_joint_state_publisher_->send(10, 999, 999, q); pump(30);
  EXPECT_EQ(diagnostic().header.frame_id, "vla:11:0:0");
  mock_joint_state_publisher_->send(11, 1, 1, position_states_); pump(30);
  EXPECT_EQ(diagnostic().header.frame_id, "vla:11:1:0");
}

TEST_F(JointImpedanceControllerTest, HoldsBeforeManagedCommandAndExposesInterfaces) {
  ASSERT_EQ(startController(), CallbackReturn::SUCCESS);
  pump(40);
  mock_joint_state_publisher_->send(0, 1, 1, position_states_); pump(40);
  EXPECT_EQ(diagnostic().header.frame_id, "vla:0:0:0");
  EXPECT_EQ(controller_->command_interface_configuration().names.size(), 7u);
  EXPECT_EQ(controller_->state_interface_configuration().names.size(), 14u);
  for (double torque : joint_commands_) EXPECT_TRUE(std::isfinite(torque));
}

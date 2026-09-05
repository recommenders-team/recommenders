# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.
#####################################################################
# Outputs the VM IP
#####################################################################
output "ip" {
  value = alicloud_instance.reco.public_ip
}

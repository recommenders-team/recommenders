# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

variable "vm_name" {
  default = "reco-vm"
}

variable "resource_group_name" {
  default = "reco"
}

variable "region" {
  # See https://help.aliyun.com/en/document_detail/40654.html
  # For Shanghai, use:
  #   default = "cn-shanghai"
  # For Shenzhen, use:
  #   default = "cn-shenzhen"
  # For Hangzhou, use:
  #   default = "cn-hangzhou"
  # For Ulanqab (cheapest), use:
  default = "cn-wulanchabu"
}

variable "instance_type_family" {
  # See https://help.aliyun.com/en/ecs/user-guide/instance-specification-naming-and-classification
  #
  # Product code
  # * 'ecs' - Elastic Compute Service (ECS)
  # Family
  # * CPU
  #   + 'ecs.e'
  #     - 'e' - Economy
  # * GPU
  #   + 'ecs.gn6i' - NVIDIA T4
  #     - 'gn' - NVIDIA GPUs
  #     - '6' - Volta/Turing
  #     - 'i' - T4 (inference)
  #   + 'ecs.gn7i' - NVIDIA A10
  #     - '7' - Ampere
  #   + 'ecs.gn8is' - NVIDIA L20
  #     - '8' - Ada Lovelace
  default = "ecs.gn8is"
}

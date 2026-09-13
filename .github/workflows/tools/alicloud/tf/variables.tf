# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

#####################################################################
# Input parameters
#####################################################################
variable "name" {
  default = "reco-vm"
}

variable "region" {
  # See https://help.aliyun.com/en/document_detail/40654.html
  # Sort by price increasingly:
  # * Outside China (No mirrors or proxies for GitHub and Docker are required.)
  #   + Jakarta,   "ap-southeast-5"
  #   + Tokyo,     "ap-northeast-1"
  #   + Frankfurt, "eu-central-1"
  #   + Singapore, "ap-southeast-1"
  # * China (Mirrors or proxies for GitHub and Docker are required.)
  #   + Ulanqab,   "cn-wulanchabu"
  #   + Hangzhou,  "cn-hangzhou"
  #   + Shanghai,  "cn-shanghai"
  #   + Shenzhen,  "cn-shenzhen"
  default = "ap-southeast-1"
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
  #     - price: ~ 11CNY / 8CNY
  #   + 'ecs.gn7i' - NVIDIA A10
  #     - '7' - Ampere
  #     - price: ~ 18CNY / 7CNY
  #   + 'ecs.gn8is' - NVIDIA L20
  #     - '8' - Ada Lovelace
  #     - price: ~ 19CNY / 3CNY
  default = "ecs.gn8is"
}

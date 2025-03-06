#!/bin/bash
sudo mkdir -p /etc/cloud/cloud.cfg.d/ && echo "network: {config: disabled}" | sudo tee -a /etc/cloud/cloud.cfg.d/99-disable-network-config.cfg
sudo cloud-init clean
sudo netplan apply

# service ssh restart
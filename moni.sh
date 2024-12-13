#!/bin/bash
mkdir -p /home/user/monitoring/prometheus

cp ./prometheus.yaml /home/user/monitoring/prometheus

sudo docker-compose up -d
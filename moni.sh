#!/bin/bash
sudo docker-compose up -d

cp -aRp ./prometheus.yaml ./prometheus
sudo chown -R 472:472 ./grafana

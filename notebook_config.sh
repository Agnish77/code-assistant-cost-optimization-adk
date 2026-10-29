#!/bin/bash -e
echo "STARTUP-SCRIPT: START"
sudo -u jupyter gsutil cp gs://qwiklabs-gcp-01-1e21e1d85ea3-bucket/code_assistant_cost_optimization.ipynb /home/jupyter/
sudo -u jupyter gsutil -m cp -r gs://qwiklabs-gcp-01-1e21e1d85ea3-bucket/code_assistant /home/jupyter/
sudo -u jupyter gsutil -m cp -r gs://qwiklabs-gcp-01-1e21e1d85ea3-bucket/assets /home/jupyter/
echo "STARTUP-SCRIPT: END"

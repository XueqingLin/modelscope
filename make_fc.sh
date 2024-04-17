sudo make -f Makefile.fc shell
sudo nohup make -f Makefile.fc command CMD="python tests/run.py --test_dir tests/pipelines" &
sudo find /mnt2/.cache/modelscope -name requirements.txt

docker login --username=fc_ai_workload_test_6@test.aliyunid.com fc-ai-workload-matrix-oversell-registry.cn-shanghai.cr.aliyuncs.com
docker build -f Makefile.fc -t fc-ai-workload-matrix-oversell-registry.cn-shanghai.cr.aliyuncs.com:test-202404171900 .

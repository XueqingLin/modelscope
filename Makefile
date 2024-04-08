WHL_BUILD_DIR :=package
DOC_BUILD_DIR :=docs/build/

# default rule
default: whl docs

.PHONY: docs
docs:
	bash .dev_scripts/build_docs.sh

.PHONY: linter
linter:
	bash .dev_scripts/linter.sh

.PHONY: test
test:
	bash .dev_scripts/citest.sh

.PHONY: whl
whl:
	python setup.py sdist bdist_wheel

.PHONY: clean
clean:
	rm -rf  $(WHL_BUILD_DIR) $(DOC_BUILD_DIR)

# 变量定义
IMAGE_NAME = registry.cn-beijing.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.1.0-py310-torch2.1.2-tf2.14.0-1.13.1
CONTAINER_NAME = modelscope-shell
VOLUME_PATH = $(shell pwd)

CMD ?= "pwd"

# 'make shell' 命令的目标
shell:
	docker run --rm -it --network=host --runtime nvidia --gpus all \
    -v ~/.cache/modelscope/hub:/root/.cache/modelscope/hub \
    --name $(CONTAINER_NAME) -v $(VOLUME_PATH):/home/admin/modelscope $(IMAGE_NAME) /bin/bash

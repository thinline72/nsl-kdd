help:
	@cat Makefile

DOCKER_FILE?=Dockerfile
SRC?=$(shell pwd)

nsl-kdd-pyspark:
	docker run -d --name nsl-kdd-pyspark -v $(SRC):/home/jovyan/work -p 8889:8888 -p 4040:4040 jupyter/pyspark-notebook

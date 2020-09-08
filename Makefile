help:
	@cat Makefile

DOCKER_FILE?=Dockerfile
SRC?=$(shell pwd)

nsl-kdd-pyspark:
	docker run -d --rm --name nsl-kdd-pyspark -v $(SRC):/home/jovyan/work -p 8889:8888 -p 4040:4040 jupyter/pyspark-notebook:dc9744740e12
	sleep 15
	docker exec nsl-kdd-pyspark jupyter notebook list

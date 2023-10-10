IMAGE=sample
HOME=/root/app
run:
	docker build -t ${IMAGE} .
in:
	nvidia-docker run --runtime=nvidia -v ${HOME}:/home/app --rm -it ${IMAGE}:latest bash

dock=sudo docker -v 2> /dev/nul
if [ $? -eq 0 ]
then
    echo Docker already installed
else
    echo Installing docker
    apt-get remove docker docker-engine docker.io containerd runc
    apt-get update
    apt-get install \
        apt-transport-https \   
        ca-certificates \
        curl \
        gnupg-agent \
        software-properties-common
    curl -fsSL https://download.docker.com/linux/debian/gpg | sudo apt-key add -
    add-apt-repository \
        "deb [arch=amd64] https://download.docker.com/linux/debian \
        $(lsb_release -cs) \
        stable"
    apt-get update
    apt-get install docker-ce docker-ce-cli containerd.io -y
fi

name='container_onnx'
onnx=0
if [[ $(sudo docker ps -a --filter "name=^/$name$" --format '{{.Names}}') != $name ]]
then
    echo Installing container_onnx
    sudo docker pull onnx/onnx-ecosystem
    git clone https://github.com/onnx/onnx-docker
    sudo docker run -it -d --name container_onnx -v /etc/localtime:/etc/localtime onnx/onnx-ecosystem
    sudo docker exec container_onnx pip install psutil
    sudo docker stop container_onnx
	onnx=1
else
    echo container_onnx already installed
fi

name='container_h5'
h5=0
if [[ $(sudo docker ps -a --filter "name=^/$name$" --format '{{.Names}}') != $name ]]
then
    echo Installing container_h5
    sudo docker pull tensorflow/tensorflow
    sudo docker run -it -d --name container_h5 -v /etc/localtime:/etc/localtime tensorflow/tensorflow 
    sudo docker exec container_h5 pip install pillow
    sudo docker exec container_h5 pip install argparse
    sudo docker exec container_h5 pip install pandas
    sudo docker exec container_h5 pip install psutil
    sudo docker stop container_h5
	h5=1
else
    echo container_h5 already installed
fi

if [ $(( $onnx + $h5 )) -gt 0 ]
then
  sudo bash ./Desktop/Raspberry-pi/scripts/container-setup.sh
fi
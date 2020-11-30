echo $1$2$3$4

if [ $1 = "-h" -o $1 = "--help" ]
then
    echo " Choose your type of model, id of model, type of test, number of test, save or not."
    echo " you can have more information on models with onnx -h or h5 -h"
else

    sudo bash ./Desktop/Raspberry-pi/scripts/folder-status.sh

    path_host="./Desktop/Raspberry-pi/resultats"
    container=""

    extension="${1##*.}"
    if [ $extension = "onnx" ]
    then
        container="container_onnx"
        path_docker="/scripts/test-model/resultats"
    fi
    if [ $extension = "h5" ]
    then
        container="container_h5"
        path_docker="/test-model/resultats"
    fi

    sudo docker start $container

    os=`sudo docker system info --format '{{.OperatingSystem}}'`
    echo $os"" with docker > $path_host""/os-name.txt
    sudo docker cp $path_host""/os-name.txt $container"":$path_docker
    sudo rm $path_host""/os-name.txt

    if [ $4 = "s" ]
    then
        sudo docker cp $path_host""/resultats.csv $container"":""$path_docker
    fi

    sudo docker exec -it $container python3 ./test-model/compare-models-""$extension"".py $1 $2 $3 $4

    if [ $4 = "s" ]
    then
        sudo docker cp $container"":""$path_docker""/resultats.csv $path_host

        sudo docker exec $container rm ./test-model/resultats/resultats.csv
    fi

    sudo docker exec $container rm ./test-model/resultats/os-name.txt

    sudo docker stop $container

fi

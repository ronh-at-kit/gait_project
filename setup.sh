#!/usr/bin/env bash
# Environmental variable def
unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=Mac;;
    CYGWIN*)    machine=Cygwin;;
    MINGW*)     machine=MinGw;;
    *)          machine="UNKNOWN:${unameOut}"
esac
echo "the OS is a : ${machine}"


if [[ -z "${GAITPATH}" ]]; then
  echo ${PWD}
  GAITPATH=${PWD}
  export GAITPATH=${GAITPATH}
else
  echo "gait path set to : ${GAITPATH}"
fi


if [ "Mac" = ${machine} ]; then
  echo '# ADDED BY GAIT PROJECT' >> ~/.bash_profile
  echo "export GAITPATH=${GAITPATH}" >> ~/.bash_profile
  echo 'export PYTHONPATH=${PYTHONPATH}:${GAITPATH}' >> ~/.bash_profile
  source ~/.bash_profile
fi


if [ "Linux" = ${machine} ]; then
  echo '# ADDED BY GAIT PROJECT' >> ~/.bashrc
  echo "export GAITPATH=${GAITPATH}" >> ~/.bashrc

  echo 'export PYTHONPATH="${PYTHONPATH}:${GAITPATH}"' >> ~/.bashrc
  source ~/.bashrc
fi


# install files
source activate torch
conda install -c menpo opencv

pip install -r requirements.txt

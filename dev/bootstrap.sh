#!/bin/bash
set -euo pipefail

# run this script using `sudo -E ./bootstrap.sh`, otherwise it won't work
#
# If you're seeing issues with `dpkg` on platforms like tensordock, wait 15
# minutes and try running the script again.
#
# Once installed to see the autoupdater logs, run 
# `sudo journalctl -fu vision-autoupdater`
################################################################################

# external vars the user may override
################################################################################
WITH_AUTOUPDATES=${WITH_AUTOUPDATES:-1}
NVIDIA_DRIVER_VERSION=${NVIDIA_DRIVER_VERSION:-535}
NO_LAUNCH=${NO_LAUNCH:-0}

# internal vars
################################################################################
REBOOT_REQUIRED=0
export DEBIAN_FRONTEND=noninteractive

function echo_() {
  echo "# $@"
  echo "################################################################################"
}

# check for root and setup exit trap
################################################################################
if [[ $(id -u) -ne 0 ]]; then
  echo_ "Please run this script as root."
  exit 1
fi

function on_exit_ {
  echo_ cleaning up...
  apt-mark unhold openssh-server
}
trap on_exit_ INT TERM EXIT

# setup base files/folders
################################################################################
echo_ setting up base files and folders
touch $HOME/.bashrc
chown $SUDO_USER:$SUDO_USER $HOME/.bashrc
chmod 644 $HOME/.bashrc

mkdir -p $HOME/.local/bin
if ! [[ $(echo $PATH | grep "$HOME/.local/bin") ]]; then
  echo '' >> $HOME/.bashrc
  echo 'export PATH=$HOME/.local/bin:$PATH' >> $HOME/.bashrc
fi
chown -R $SUDO_USER:$SUDO_USER $HOME/.local

# do not upgrade openssh server whilst installing
################################################################################
apt-mark hold openssh-server

# fix anything broken, update stuff, and install base software
################################################################################
echo_ setting up base packages
apt update -qq
apt install -y vim git curl wget cron net-tools dnsutils software-properties-common

# docker
################################################################################
echo_ checking for docker
if ! [[ $(which docker) ]]; then
  echo_ docker was not found, installing...
  apt update -qq
  apt install -y docker.io
  systemctl enable --now docker
fi

groupadd docker || true
usermod -aG docker $SUDO_USER || true

# docker compose
################################################################################
echo_ "checking for docker-compose"
if ! [[ $(which docker-compose) ]]; then
  echo_ "docker-compose was not found, installing..."
  apt-get update
  apt-get install -y docker-compose-plugin
fi

groupadd docker || true
usermod -aG docker $SUDO_USER || true

# nvidia drivers
################################################################################
echo_ checking for nvidia drivers
if ! [[ $(which nvidia-smi) ]]; then
  echo_ nvidia drivers were not found, installing...

  apt purge nvidia-*
  add-apt-repository -y ppa:graphics-drivers/ppa
  apt update -qq
  apt-mark unhold nvidia* libnvidia*
  apt install -y libnvidia-common-$NVIDIA_DRIVER_VERSION libnvidia-gl-$NVIDIA_DRIVER_VERSION nvidia-driver-$NVIDIA_DRIVER_VERSION
  dpkg-query -W --showformat='${Package} ${Status}\n' | grep -v deinstall | awk '{ print $1 }' | grep -E 'nvidia.*-[0-9]+$' | xargs -r -L 1 apt-mark hold

  # signal to the script that it _should_ reboot the server once done.
  REBOOT_REQUIRED=1
fi

# nvidia/cuda/container-toolkit
################################################################################
echo_ checking for the nvidia container toolkit
if ! [[ $(command nvidia-ctk) ]]; then
  echo_ the nvidia container toolkit was not found, installing...
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
  dpkg -i cuda-keyring_1.1-1_all.deb
  rm cuda-keyring_1.1-1_all.deb

  apt update -qq
  apt-mark unhold nvidia* libnvidia*
  apt install -y nvidia-container-toolkit
  dpkg-query -W --showformat='${Package} ${Status}\n' | grep -v deinstall | awk '{ print $1 }' | grep -E 'nvidia.*-[0-9]+$' | xargs -r -L 1 apt-mark hold
fi

nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# python 3.12
################################################################################
echo_ checking for python3.12
if ! [[ $(which python3.12) ]]; then
  echo_ python3.12 was not found, installing...
  add-apt-repository -y ppa:deadsnakes/ppa
  apt install -y python3.12-full
  python3.12 -m ensurepip
fi

# ensure `python` and `pip` point to the right place
rm /usr/bin/pip || true
echo '#!/bin/bash' >> /usr/bin/pip
echo 'python3.12 -m pip $@' >> /usr/bin/pip
chmod a+x /usr/bin/pip

rm /usr/bin/python || true
ln -s $(which python3.12) /usr/bin/python

# poetry
################################################################################
echo_ checking for poetry
if ! [[ $(which poetry) ]]; then
  echo_ installing poetry
  curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.7.1 python3.12 -
  
  # Add poetry to PATH if not already present
  if ! grep -q "export PATH=\"/root/.local/bin:\$PATH\"" ~/.bashrc; then
    echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc
  fi
  source ~/.bashrc

  # Configure poetry settings
  poetry config virtualenvs.in-project true
  poetry config installer.parallel true

  # Create directories if they don't exist and fix permissions
  mkdir -p $HOME/.local/share/pypoetry
  mkdir -p $HOME/.config/pypoetry
  chown -R $SUDO_USER:$SUDO_USER $HOME/.local/share/pypoetry
  chown -R $SUDO_USER:$SUDO_USER $HOME/.config/pypoetry
else
  echo_ poetry is already installed
fi

# configure servers to start on boot
################################################################################
if [[ NO_LAUNCH -eq 1 ]]; then
  :
else
  if [[ WITH_AUTOUPDATES -eq 1 ]]; then
    sudo -E bash run_autoupdater.sh
  else
    # just schedule `launch_orchestrator.sh` to run periodically
    if ! [[ $(crontab -u $SUDO_USER -l | grep -F 'launch_orchestrator.sh') ]]; then
      (crontab -u $SUDO_USER -l; echo "* * * * * /bin/bash -c 'source $HOME/.bashrc; cd vision-workers && bash launch_orchestrator.sh >> vision_orchestrator.log 2>&1'") | crontab -u $SUDO_USER -
    fi
  fi
fi


# finally, reboot if needed
################################################################################
if [[ REBOOT_REQUIRED -eq 1 ]]; then
  echo_ "bootstrap.sh modified something that requires a reboot. Please SSH back in to this machine after a short while :)"
  shutdown now -r
else
  echo_ "bootstrap.sh is all done :)"
fi
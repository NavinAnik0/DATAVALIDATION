#! /bin/bash

basic_setup()
{
  sudo apt-get update
  sudo apt-get upgrade
  sudo apt-get install zip unzip

  curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
  unzip awscliv2.zip
  sudo ./aws/install

  sudo apt-get upgrade -y linux-aws
  sudo reboot
}

install_cuda()
{
  sudo apt install python3-pip
  sudo apt-get install -y gcc make linux-headers-$(uname -r)
  echo " blacklist vga16fb" | sudo tee --append /etc/modprobe.d/blacklist.conf
  echo " blacklist nouveau" | sudo tee --append /etc/modprobe.d/blacklist.conf
  echo " blacklist rivafb" | sudo tee --append /etc/modprobe.d/blacklist.conf
  echo " blacklist nvidiafb" | sudo tee --append /etc/modprobe.d/blacklist.conf
  echo " blacklist rivatv" | sudo tee --append /etc/modprobe.d/blacklist.conf
  echo 'GRUB_CMDLINE_LINUX="rdblacklist=nouveau"' | sudo tee --append etc/default/grub
  sudo update-grub
  aws s3 cp --recursive s3://ec2-linux-nvidia-drivers/latest/ .
  chmod +x NVIDIA-Linux-x86_64*.run
  sudo /bin/sh ./NVIDIA-Linux-x86_64*.run
  sudo reboot
}


if [[ "$1" = "setup" ]]
then
  basic_setup
fi

if [[ "$1" = "install" ]]
then
  install_cuda
fi